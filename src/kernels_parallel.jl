#kernels_parallel.jl

function hpl{T}(A::DArray{T,2}, b)
    F = lufact!(Array(A)) #Partial pivoting on by default
    F \ b
end

#Look up which processor owns index idx in DArray
function procof{T}(A::DArray{T,1}, idx::Integer)
    for (iproc, idxs) in enumerate(A.indexes)
        localidx = 0
        for ir in idxs, id in ir
            localidx += 1
            if idx==id
                return A.pids[iproc], localidx
            end
        end
    end
end

updatearr(A, i, r) = (localpart(A)[i] $= r)

function randomupdate!{T<:Integer}(A::DArray{T,1}, nupdate)
    m = size(A, 1)
    for i=1:nupdate
        r = rand(T)
        index = r & (m-1) + 1
        p, i = procof(A, index)
        remotecall_wait(updatearr, p, A, i, r)
    end
end

function streamtriad!{T}(a::DArray{T,1}, b::DArray{T,1}, α::T, c::DArray{T,1}, ntrial=10)
    @assert a.cuts==b.cuts==c.cuts "Cuts are not aligned"
    m = size(a, 1)
    for j=1:ntrial
        @sync for p in a.pids
            @async remotecall_fetch(()->(streamtriad!(localpart(a), localpart(b), α, localpart(c))), p)
        end
    end
end

FFT(z::DArray) = fft!(Array(z))
