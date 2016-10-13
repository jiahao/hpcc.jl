#IMPLEMENTATION OF THE FOUR KERNELS
#Reference: http://www.hpcchallenge.org/class2specs.pdf

function hpl(A, b)
    F = lufact(A) #Partial pivoting on by default
    A \ b
end

function randomupdate!{T}(A::Vector{T}, nupdate)
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

################################################################################

#Helper routines that initialize, execute, time, and validate the kernels

function runhpl(n)
    #Initialize
    hpl(rand(1, 1), rand(1)) #Precompile
    A = randn(n, n)
    b = randn(n)

    #Run
    t = @elapsed x=hpl(A, b)

    #Validate
    r = b - A*x
    ϵ = eps()
    r₀= norm(r,Inf)
    nrmA1 = norm(A, 1)
    r₁= r₀ / (ϵ*nrmA1*n)
    r₂= r₀ / (ϵ*nrmA1*norm(x,1))
    r₃= r₀ / (ϵ*norm(A,Inf)*norm(x,Inf)*n)
    err = max(r₁, r₂, r₃)
    if err ≥ 16
        warn("Error", err, "exceeds allowed value of 16")
    end
    return t
end

function runrandomupdate(m, nupdate=4m, seed=1)
    #Initialize
    randomupdate!(UInt64[1], 1) #Precompile
    srand(seed)
    T = Array{UInt64}(m)
    for i = 1:m
        T[i] = i
    end

    #Run
    t = @elapsed randomupdate!(T, nupdate)

    #Validate
    srand(seed)
    randomupdate!(T, nupdate)
    err = 0
    for i = 1:m
        err += (T[i] != i)
    end
    err /= m
    if err > 0.01
        warn("Error rate = ", 100err, "%, exceeding 1% allowance")
    end

    return t
end

function runstreamtriad(m, ntrials=10)
    #Initialize
    streamtriad!([0.0], [1.0], 2.0, [3.0])
    a = zeros(m)
    b = randn(m)
    c = randn(m)
    α = randn()

    #Run
    t = 0.0
    for i in 1:ntrials
        t = min(t, @elapsed streamtriad!(a, b, α, c))
    end
    return t
end

function runfft(m)
    #Initialize
    FFT([1.0+0.0im])
    z = Vector{Complex128}(m)
    for i in 1:m
        z[i] = randn() + im*randn()
    end
    z′ = copy(z)

    #Run
    t = @elapsed FFT(z)

    #Validate
    ifft!(z)
    err = maxabs(z′ - z)
    errtol = 16*eps()*log2(m)
    if err > errtol
        warn("Error = $err exceeds tolerance of $errtol")
    end

    return t
end

################################################################################

#Run the tests

println("Serial HPL")
for e in 3:0.2:3.8
    n = round(Int, 10^e)
    println("Problem size = ", n)
    t = runhpl(n)
    println("Run time = $t s")
    println("Gigaflops = ", 1e-9*(2n^3/3 + 3n^2/2)/t)
end

println()
println()
println()
println("Serial random update")
for e in 18:24
    m = 2^e
    println("Problem size = ", m)
    t = runrandomupdate(m)
    println("Run time = $t s")
    println("GUPS = ", m*1e-9/t)
end
exit()
println()
println()
println()
println("Serial STREAM triad")
for e in 6.5:0.5:8.0
    m = round(Int, 10^e)
    println("Problem size = ", m)
    t = runstreamtriad(m)
    println("Run time = $t s")
    println("Gigabytes/sec = ", 2.4e-8*10*m/t)
end

println()
println()
println()
println("Serial FFT")
for e in 6:0.5:8
    m = round(Int, 10^e)
    println("Problem size = ", m)
    t = runfft(m)
    println("Run time = $t s")
    println("Gigaflops = ", 5e-9*m*log2(m)/t)
end
