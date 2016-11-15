#Communication avoiding LU

import Base: start, next, done, size, getindex
import DistributedArrays: DArray

#              |
#   A(ma x na) | B(ma x nb)
# -------------+------------
#   C(mb x na) | D(mb x nb)

immutable RowBlockedMatrix{T,S<:AbstractMatrix} <: AbstractMatrix{T}
    A::S
    m::Int
    n::Int
    k::Int
end

size(A::RowBlockedMatrix) = ceil(Int, A.m/A.k), 1
size(A::RowBlockedMatrix, i::Int) = if i==1 ceil(Int, A.m/A.k) elseif i==2 1 else error() end

function getindex(A::RowBlockedMatrix, i, j=1)
    rowstart = (i-1)*A.k + 1
    rowend = min(i*A.k, A.m)
    view(A.A, rowstart:rowend, 1:A.n)
end

RowBlockedMatrix{T}(A::AbstractMatrix{T}, k) = RowBlockedMatrix{T,typeof(A)}(A, size(A, 1), size(A, 2), k)

blockrows(A, k) = RowBlockedMatrix(A, k)

#RowBlockedMatrix is iterable over blocked rows
start(A::RowBlockedMatrix) = 1
next(A::RowBlockedMatrix, iblock) = (A[iblock], iblock+1)
done(A::RowBlockedMatrix, iblock) = iblock == 1+ceil(Int, A.m/A.k)

function whoowns{T,N}(A::SubArray{T,N,DArray{T,2,Matrix{T}}})
    rowrange = A.indexes[1]
    colrange = A.indexes[2]

    #Find out which pid owns the most data
    myid = myndata = 0
    paridx = A.parent.indexes
    for i = 1:size(paridx, 1), j = 1:size(paridx, 2)
        localrowrange = rowrange ∩ paridx[i,j][1]
        localcolrange = colrange ∩ paridx[i,j][2]

        ndata = length(localrowrange)*length(localcolrange)
        if ndata > myndata
            myid = A.parent.pids[i,j]
            myndata = ndata
        end
    end
    #info("whoown darray: $myid ($(myndata/(length(rowrange)*length(colrange))))")
    return myid
end



#Tall and skinny LU
#k - row block size
#b - number of pivot rows
function tslu!(A, piv, k, b)
    #Step 1: Find set of good pivot rows
    #Serial LU on each block row
    #t = time()
    #info("Time $(round(time() - t, 3)): Step 1: broadcast")
    lus = []
    for blockrow in blockrows(A, k)
        push!(lus, @spawnat whoowns(blockrow) lufact(Array(blockrow)))
    end
    #info("Time $(round(time() - t, 3)): Step 1: collect")
    lus = map(fetch, lus)
    for l in lus
        if isa(l,RemoteException)
            showerror(STDERR, l)
            error()
        end
    end
    #info("Time $(round(time() - t, 3)): Step 1: collect done")
    while length(lus) > 1
        #info("Time $(round(time() - t, 3)): Step 1: recursion with $(length(lus)) work to do")
        newlus = []
        nlu = length(lus)
        npairs, isodd = divrem(nlu, 2)
        for ipair in 1:npairs
            Upair = [lus[2ipair-1][:U]; lus[2ipair][:U]]
            #push!(newlus, @spawnat(@show(whoowns(blockrow)), lufact(blockrow)))
            push!(newlus, lufact(Upair))
        end
        if isodd==1
            #push!(newlus, @spawnat(@show(whoowns(blockrow)), lufact(blockrow)))
            push!(newlus, lus[end])
        end
        lus = newlus
    end
    #info("Time $(round(time() - t, 3)): Step 1: recursion done")

    #Step 2: Permute pivot rows into first b rows of the panel
    #info("Time $(round(time() - t, 3)): Step 2: pivot")
    perm = view(LinAlg.ipiv2perm(lus[1][:p], size(A, 2)), 1:min(b, size(A, 2)))
    permuterows!(A, perm)
    #info("Time $(round(time() - t, 3)): Step 2: pivot done")

    #Step 3: Unpivoted LU on panel
    #info("Time $(round(time() - t, 3)): Step 3: LU on panel")
    F = if b ≥ size(A, 2)
        lufact!(A, Val{false})
    else
        lufact!(view(A, :, 1:b), Val{false})
    end
    #info("Time $(round(time() - t, 3)): Step 3: LU done")
end

function pidmap{T}(A::SubArray{T,2,DArray{T,2,Matrix{T}}})
    pids = Dict()

	#XXX use dispatch to make the subrowrange, subcolrange computations cleaner
    subrowrange = A.indexes[1]
    if isa(subrowrange, Colon)
         subrowrange = 1:size(A, 1)
    elseif isa(subrowrange, Integer)
         subrowrange = subrowrange:subrowrange
    end

    subcolrange = A.indexes[2]
    if isa(subcolrange, Colon)
        subcolrange = 1:size(A, 2)
    elseif isa(subcolrange, Integer)
        subcolrange = subcolrange:subcolrange
    end

    for (p, (localrowrange, localcolrange)) in zip(A.parent.pids, A.parent.indexes)
        #Compute intersection in global address space
        rows = subrowrange ∩ localrowrange
        cols = subcolrange ∩ localcolrange
        if length(rows) > 0 && length(cols) > 0
            #Compute blocks in process-local address space
            localrows = rows-localrowrange.start+1
            localcols = cols-localcolrange.start+1

            subrows = rows-subrowrange.start+1
            subcols = cols-subcolrange.start+1

            pids[p] = rows, cols, localrows, localcols, subrows, subcols

            #@assert A.parent[rows, cols] ==
            #fetch(@spawnat p localpart(A.parent)[localrows, localcols]) ==
            #A[subrows, subcols] #Not implemented
        end
    end
    pids
end



function Base.lufact!{T,piv}(
        A::SubArray{T,2,DArray{T,2,Matrix{T}}},
        ::Type{Val{piv}}
    )

    #Copy to local memory
    tmpA = Array(A)

    #Do computation locally
    lufact!(tmpA, Val{piv})

    #Redistribute results
    pids = pidmap(A)
    @sync for (p, (gr, gc, lr, lc, sr, sc)) in pids
        @spawnat p localpart(A.parent)[lr, lc] = view(tmpA, sr, sc)
    end
end

function Base.A_ldiv_B!{T}(
        F::LinAlg.LU{T, DArray{T, 2, Array{T,2}}},
        b::AbstractVector{T}
    )

    Ff = Array(F.factors)
    U = UpperTriangular(Ff)
    L = LinAlg.UnitLowerTriangular(Ff)
    bp = b[LinAlg.ipiv2perm(F.ipiv, length(b))]
    x = L \ bp
    return U \ x
end

#Intesection of index ranges
function Base.intersect(A::Tuple{UnitRange{Int64},UnitRange{Int64}},
                        B::Tuple{UnitRange{Int64},UnitRange{Int64}})
    a1, a2 = A
    b1, b2 = B
    return (a1∩b1, a2∩b2)
end

function DistributedArrays.localpart{T,N}(A::SubArray{T,N,DArray{T,2,Matrix{T}}})
    subindexes = A.indexes
    local globallocalindex, globalsubindex
    for (p, localindexes) in zip(A.parent.pids, A.parent.indexes)
        if p == myid()
            #Compute intersection of indexes in global address space
            globallocalindex = localindexes
            globalsubindex = (localindexes ∩ subindexes)
            break
        end
    end

    #Convert local index from global address space to process-local address space
    localindex1 = globalsubindex[1] - globallocalindex[1].start + 1
    localindex2 = globalsubindex[2] - globallocalindex[2].start + 1
    return view(localpart(A.parent), localindex1, localindex2)
end

#Swap rows of a matrix in place
function swaprows!(A::AbstractMatrix, r1, r2)
    nc = size(A, 2)
    for i=1:nc
        A[r1, i], A[r2, i] = A[r2,i], A[r1,i]
    end
end

#Swap rows of a DArray in place
# e.g. A = drandn(4000, 4000)
#      swaprows!(A, [(1,2), (1000,1001)])
# swaps the first two rows of A in place, and also rows 1000 and 1001
function swaprows!{T}(A::SubArray{T,2,DArray{T,2,Matrix{T}}}, r1, r2)
    pidmap1 = pidmap(view(A, r1:r1, :))
    pidmap2 = pidmap(view(A, r2:r2, :))

    for (i,((p1, (gr1, gc1, lr1, lc1, sr1, sc1)),
         (p2, (gr2, gc2, lr2, lc2, sr2, sc2)))) in enumerate(zip(pidmap1, pidmap2))

        #Need to look up which parts in pidmap2 align
        # info("FROM proc $p1, chunk $gr1,$gc1 ($lr1, $lc1) $sr1, $sc1")
        # info("TO   proc $p2, chunk $gr2,$gc2 ($lr2, $lc2) $sr2, $sc2")
            if p1 == p2 #Do all the work locally
                # info("Local work")
                @assert lc2 == lc2
                @sync @spawnat p1 swaprows!(localpart(A), sr1, sr2)
            else #copy from remote, swap, send
                # info("Local work $p2 -> $p1")
                @sync @spawnat p1 begin
                     lA = localpart(A)
                     B = Array(view(A, sr1, sc1))

                     for (i1, i2) in enumerate(sr1), j in sr2
                         B[i1, j], lA[i2, j] = lA[i2, j], B[i1, j]
                     end

                     @spawnat p2 localpart(A)[sr2, sc2] = B
                end
        end
    end
    #info("swaprows! done")
end

function permuterows!{T}(A::SubArray{T,2,DArray{T,2,Matrix{T}}}, perm)
    # info("permuterows! with permutation $perm")
    for (a, b) in enumerate(perm)
        if a!=b
            swaprows!(A, a, b)
        end
    end
end

function calu!(A, k=64, b=64)
    m, n = size(A)
    piv = collect(1:m)
    for i = 1:ceil(Int, n/k)
        #info("calu!: tslu! on [$((1+(i-1)*k):m), $((1+(i-1)*k):n)]")
        tslu!(view(A, (1+(i-1)*k):m, (1+(i-1)*k):n), piv, k, b)
    end
    LinAlg.LU(A, piv, 0)
end
