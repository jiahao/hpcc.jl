#kernels_naive.jl

function hpl(A, b)
    F = lufact!(A) #Partial pivoting on by default
    F \ b
end

function randomupdate!{T<:Integer}(A::AbstractVector{T}, nupdate)
    m = size(A, 1)
    for i=1:nupdate
        r = rand(T)
        index = r & (m-1) + 1
        A[index] $= r
    end
end

function streamtriad!(a, b, α, c, ntrial=10)
    m = size(a, 1)
    for j=1:ntrial, i=1:m
        a[i] = b[i] + α*c[i]
    end
end

FFT(z) = fft!(z)
