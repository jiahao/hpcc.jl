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
        push!(lus, @spawnat(@show(whoowns(blockrow)), lufact(Array(blockrow))))
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
    F = lus[1]
    permuterows!(A, F[:p])

    #Step 3: Unpivoted LU on panel
    F = if b ≥ size(A, 2)
        lufact!(A, Val{false})
    else
        lufact!(view(A, :, 1:b), Val{false})
    end
end



function pidmap{T}(A::SubArray{T,2,DArray{T,2,Matrix{T}}})
    pids = Dict()
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



function Base.lufact!{T}(A::SubArray{T,2,DArray{T,2,Matrix{T}}}, ::Type{Val{false}})

    pids = pidmap(A)
    #Copy to local memory
    tmpA = Array(T, size(A))
    for (p, (gr, gc, lr, lc, sr, sc)) in pids
        tmpA[sr, sc] = fetch(@spawnat p view(localpart(A.parent), lr, lc))
    end

    #Do computation locally
    lufact!(tmpA, Val{false})

    #Redistribute results
    tmpA = Array(T, size(A))
    for (p, (gr, gc, lr, lc, sr, sc)) in pids
        @spawnat p localpart(A.parent)[lr, lc] = tmpA[sr, sc]
    end
end



function Base.A_ldiv_B!{T}(F::LinAlg.LU{T, DArray{T, 2, Array{T,2}}},
                      b::AbstractVector{T})

    UpperTriangular(F.factors) \ (
      LinAlg.UnitLowerTriangular(F.factors) \ (
        b[LinAlg.ipiv2perm(F.ipiv, length(b))]
    ))
end



#permuterows!(A, perm) = (info(typeof(A)); A[:,:] = A[perm, :])
function permuterows!{T}(A::SubArray{T,2,DArray{T,2,Matrix{T}}}, perm)
    println("permute $perm")
    swaps = Dict()
    pids = Dict()
    for (i, r) in enumerate(perm)
        if i!=r
            p1 = pids[i] = get(pids, i, whoowns(view(A,i,:)))
            p2 = pids[r] = get(pids, r, whoowns(view(A,r,:)))

            if p1 < p2
                swaps[(p1, p2)] = push!(get(swaps, (p1, p2), []), (i, r))
            else
                swaps[(p2, p1)] = push!(get(swaps, (p2, p1), []), (r, i))
            end
        end
    end

    if length(swaps) == 0 #nothing to do
        return
    end

    info("Need to do some swaps")
    tasklist = []

    @everywhere function swaprows!(A, swaplist)
        for (r1, r2) in swaplist
            pidmap1 = pidmap(sub(A, r1:r1, :))
            pidmap2 = pidmap(sub(A, r2:r2, :))
            info(pidmap1)
            info(pidmap2)

           #for (i, j) in list, k in 1:size(A, 2)
            #A[i, k], A[j, k] = A[j, k], A[i, k]
        end
    end

    for ((p1, p2), list) in swaps
        println("$p1->$p2 must do $(length(list))")
        if p1 == p2
            dump(A)
            push!(tasklist, @spawnat p1 swaprows!(A, list))
        else
            #push!(tasklist, @spawnat p2 swaprows!(A, list, p1))
        end
    end
    map(fetch, tasklist)
    error("Not impleented")
end


function calu!(A, k=64, b=64)
    m, n = size(A)
    piv = collect(1:m)
    for i = 1:ceil(Int, n/k)
        tslu!(view(A, (1+(i-1)*k):m, (1+(i-1)*k):n), piv, k, b)
    end
    LinAlg.LU(A, piv, 0)
end

