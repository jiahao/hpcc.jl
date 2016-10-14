#kernels_naive.jl

function hpl(A, b)
    F = lufact!(A) #Partial pivoting on by default
    F \ b
end

function randomupdate!{T<:Integer}(A::AbstractVector{T}, rnd::AbstractVector{T})
    m = size(A, 1)
    for r in rnd
        index = r & (m-1) + 1
        A[index] $= r
    end
end

function streamtriad!(a, b, α, c, ntrial=10)
    m = size(a, 1)
    for j=1:ntrial, i in eachindex(a)
        a[i] = b[i] + α*c[i]
    end
end

FFT(z) = fft!(z)
