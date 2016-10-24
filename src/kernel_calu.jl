#Communication avoiding LU

import Base: start, next, done, size, getindex

#              |
#   A(ma x na) | B(ma x nb)
# -------------+------------
#   C(mb x na) | D(mb x nb)

type RowBlockedMatrix{T} <: AbstractMatrix{T}
    A
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

RowBlockedMatrix{T}(A::AbstractMatrix{T}, k) = RowBlockedMatrix{T}(A, size(A, 1), size(A, 2), k)

blockrows(A, k) = RowBlockedMatrix(A, k)

#RowBlockedMatrix is iterable over blocked rows
start(A::RowBlockedMatrix) = 1
function next(A::RowBlockedMatrix, iblock)
    A[iblock], iblock+1
end
done(A::RowBlockedMatrix, iblock) = iblock == 1+ceil(Int, A.m/A.k)

#Tall and skinny LU
#k - row block size
#b - number of pivot rows
function tslu!(A, piv, k, b)
    #Step 1: Find set of good pivot rows
    #Serial LU on each block row
    lus = []
    for blockrow in blockrows(A, k)
        push!(lus, lufact(blockrow))
    end

    while length(lus) > 1
        newlus = []
        nlu = length(lus)
        npairs, isodd = divrem(nlu, 2)
        for ipair in 1:npairs
            Upair = [lus[2ipair-1][:U]; lus[2ipair][:U]]
            push!(newlus, lufact(Upair))
        end
        if isodd==1
            push!(newlus, lus[end])
        end
        lus = newlus
    end

    #Step 2: Permute pivot rows into first b rows of the panel
    F = lus[1]
    A = A[F[:p], :] #Permute everything - very much overkill

    #Step 3: Unpivoted LU on panel
    F = if b ≥ size(A, 2)
        lufact!(A, Val{false})
    else
        lufact!(view(A, :, 1:b), Val{false})
    end
end

function calu!(A, k=16, b=16)
    m, n = size(A)
    piv = collect(1:m)
    for i = 1:ceil(Int, n/k)
        tslu!(view(A, (1+(i-1)*k):m, (1+(i-1)*k):n), piv, k, b)
    end
    LinAlg.LU(A, piv, 0)
end

