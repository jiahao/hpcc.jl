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

function streamtriad!{T}(a::DArray{T,1}, b::DArray{T,1}, α::T, c::DArray{T,1}, ntrial=10)
    m = size(a, 1)
    for j=1:ntrial, i in localindexes(a)
        a[i] = b[i] + α*c[i]
    end
end

FFT(z::DArray) = fft!(Array(z))
