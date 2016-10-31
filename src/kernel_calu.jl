#Communication avoiding LU

import Base: start, next, done, size, getindex

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
    t = time()
    info("Time $(round(time() - t, 3)): Step 1: broadcast")
    lus = []
    for blockrow in blockrows(A, k)
        push!(lus, @spawnat whoowns(blockrow) lufact(Array(blockrow)))
    end
    info("Time $(round(time() - t, 3)): Step 1: collect")
    lus = map(fetch, lus)
    for l in lus
        if isa(l,RemoteException)
            showerror(STDERR, l)
            exit()
        end
    end
    info("Time $(round(time() - t, 3)): Step 1: collect done")
    while length(lus) > 1
        info("Time $(round(time() - t, 3)): Step 1: recursion with $(length(lus)) work to do")
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
    info("Time $(round(time() - t, 3)): Step 1: recursion done")

    #Step 2: Permute pivot rows into first b rows of the panel
    info("Time $(round(time() - t, 3)): Step 2: pivot")
    perm = view(LinAlg.ipiv2perm(lus[1][:p], size(A, 2)), 1:min(b, size(A, 2)))
    permuterows!(A, perm)
    info("Time $(round(time() - t, 3)): Step 2: pivot done")

    #Step 3: Unpivoted LU on panel
    info("Time $(round(time() - t, 3)): Step 3: LU on panel")
    F = if b ≥ size(A, 2)
        lufact!(A, Val{false})
    else
        lufact!(view(A, :, 1:b), Val{false})
    end
    info("Time $(round(time() - t, 3)): Step 3: LU done")
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

    UpperTriangular(F.factors) \ (
      LinAlg.UnitLowerTriangular(F.factors) \ (
        b[LinAlg.ipiv2perm(F.ipiv, length(b))]
    ))
end



function DistributedArrays.localpart{T}(A::SubArray{T,N,DArray{T,2,Matrix{T}}})
    pids = pidmap(A)
    for (p, i) in pids
        if p == myid()
        end
    end
    error("Not implemented")
end



#Swap rows of a DArray in place
# e.g. A = drandn(4000, 4000)
#      swaprows!(A, [(1,2), (1000,1001)])
# swaps the first two rows of A in place, and also rows 1000 and 1001
function swaprows!{T}(A::SubArray{T,2,DArray{T,2,Matrix{T}}}, swaplist)
    @sync for (r1, r2) in swaplist
        pidmap1 = pidmap(view(A, r1:r1, :))
        pidmap2 = pidmap(view(A, r2:r2, :))

        for (p1, i1) in pidmap1
            #Need to look up which parts in pidmap2 align
            info("On proc $p1, chunk $i1")

            pidmap2 = pidmap(view(A, r2:r2, i1[2]))
            @assert length(pidmap2)==1 "Distribution not supported"

            for (p2, i2) in pidmap2
                info("MAP TO proc $p2, chunk $i2")
                if p1 == p2 #Do all the work locally
                     @spawnat p1 begin
                         lA = localpart(A)
                         info(zip(i1[3], i2[3]))
                         info(zip(i1[4], i2[4]))
                         info(size(lA))
                         info(typeof(A))
                         info(typeof(lA))
                         for (r1, r2) in zip(i1[3], i2[3]), (c1, c2) in zip(i1[4], i2[4])
                             lA[r1, c1], lA[r2, c2] = lA[r2, c2], lA[r1, c1]
                         end
                     end
                else #copy from remote, swap, send
                    @spawnat p1 begin
                         lA = localpart(A)
info("$p2 -> $p1 -> $p2")
                         info(i1)
                         info(i2)
                         info(size(lA))
                         B = Array(view(A,i2[1], i2[2]))

                         for (i1, i2) in enumerate(i1[3]), j in i1[4]
                             B[i1, j], lA[i2, j] = lA[i2, j], B[i1, j]
                         end

                         @spawnat p2 begin
                             localpart(A)[i2[3], i2[4]] = B
                         end
                    end
                end
            end
        end
    end
end



function permuterows!{T}(A::SubArray{T,2,DArray{T,2,Matrix{T}}}, perm)
    info("permuterows! with permutation $perm")
    for (a, b) in enumerate(perm)
        if a!=b
            swaprows!(A, [(a, b)])
        end
    end
end


function calu!(A, k=64, b=64)
    m, n = size(A)
    piv = collect(1:m)
    for i = 1:ceil(Int, n/k)
        info("calu!: tslu! on [$((1+(i-1)*k):m), $((1+(i-1)*k):n)]")
        tslu!(view(A, (1+(i-1)*k):m, (1+(i-1)*k):n), piv, k, b)
    end
    LinAlg.LU(A, piv, 0)
end

