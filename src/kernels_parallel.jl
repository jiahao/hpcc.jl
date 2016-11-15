#kernels_parallel.jl

include("kernel_calu.jl")

function hpl{T}(A::DArray{T,2}, b, k, l)
    F = calu!(A, k, l)
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

function updatearray!(A, w)
    Al = localpart(A)
    for (i, r) in w
	Al[i] $= r
    end
end

function randomupdate!{T<:Integer}(A::DArray{T,1}, nupdate)
    m = size(A, 1)
    work = Dict()
    for i=1:nupdate
        r = rand(T)
        index = r & (m-1) + 1
        p, i = procof(A, index)
	work[p] = push!(get(work, p, Tuple{UInt64,UInt64}[]), (i, r))
    end
    @sync begin
	for (p, w) in work
            remotecall(updatearray!, p, A, w)
	end
    end
end

function streamtriad!{T}(a::DArray{T,1}, b::DArray{T,1}, α::T, c::DArray{T,1})
    @assert a.cuts==b.cuts==c.cuts "Cuts are not aligned"
    m = size(a, 1)
    @sync for p in a.pids
        #Slightly more efficient that the alternative
        #@async remotecall_fetch(()->(streamtriad!(localpart(a), localpart(b), α, localpart(c))))
        # Ref: https://github.com/JuliaLang/julia/blob/4ba21aa57f57b9e42eeba5886d9dec772281e9f2/base/multi.jl#L100-L126
        @async remotecall_fetch((a′, b′, α′, c′)->(streamtriad!(localpart(a′), localpart(b′), α′, localpart(c′))), p, a, b, α, c)
    end
end

FFT(z::DArray) = fft!(Array(z))
