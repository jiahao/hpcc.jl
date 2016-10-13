#simple.jl

include("kernels_naive.jl")
include("kernels_pure.jl")

################################################################################

#Helper routines that initialize, execute, time, and validate the kernels

function runhpl(n)
    #Initialize
    hpl(rand(1, 1), rand(1)) #Precompile
    A = randn(n, n)
    b = randn(n)
    A′= copy(A)

    #Run
    t = @elapsed x=hpl(A, b)

    #Validate
    r = b - A′*x
    ϵ = eps()
    r₀= norm(r,Inf)
    nrmA1 = norm(A′, 1)
    r₁= r₀ / (ϵ*nrmA1*n)
    r₂= r₀ / (ϵ*nrmA1*norm(x,1))
    r₃= r₀ / (ϵ*norm(A′,Inf)*norm(x,Inf)*n)
    err = max(r₁, r₂, r₃)
    if err ≥ 16
        warn("Error", err, "exceeds allowed value of 16")
    end
    return t
end

function runhpl2(n)
    #Initialize
    hpl2(rand(1, 1), rand(1)) #Precompile
    A = randn(n, n)
    b = randn(n)
    A′= copy(A)

    #Run
    t = @elapsed x=hpl2(A, b)

    #Validate
    r = b - A′*x
    ϵ = eps()
    r₀= norm(r,Inf)
    nrmA1 = norm(A′, 1)
    r₁= r₀ / (ϵ*nrmA1*n)
    r₂= r₀ / (ϵ*nrmA1*norm(x,1))
    r₃= r₀ / (ϵ*norm(A′,Inf)*norm(x,Inf)*n)
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
    t = Inf
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
#        1234567890123456789012345678901234567890123456
println("Problem size (n) | Run time (sec.) | Gigaflops")
for e in 3:0.2:3.8
    n = round(Int, 10^e)
    @printf("%16d | ", n)
    t = runhpl(n)
    @printf("%15.6f | %15.9f\n", t, 1e-9*(2n^3/3 + 3n^2/2)/t)
end

println("\n\n\nSerial HPL v2 - pure Julia kernel")
#        1234567890123456789012345678901234567890123456
println("Problem size (n) | Run time (sec.) | Gigaflops")
for e in 3:0.2:3.4
    n = round(Int, 10^e)
    @printf("%16d | ", n)
    t = runhpl2(n)
    @printf("%15.6f | %15.9f\n", t, 1e-9*(2n^3/3 + 3n^2/2)/t)
end

println("\n\n\nSerial random update")
#        12345678901234567890123456789012345678901234567890123
println("Problem size (m) | Run time (sec.) | Gigaupdates/sec.")
for e in 18:24
    m = 2^e
    @printf("%16d | ", m)
    t = runrandomupdate(m)
    @printf("%15.6f | %15.9f\n", t, m*1e-9/t)
end

println("\n\n\nSerial STREAM triad")
#        123456789012345678901234567890123456789012345678901
println("Problem size (m) | Run time (sec.) | Gigabytes/sec.")
for e in 6.5:0.5:8.0
    m = round(Int, 10^e)
    @printf("%16d | ", m)
    t = runstreamtriad(m)
    @printf("%15.6f | %15.9f\n", t, 1e-9*24*10*m/t)
end

println("\n\n\nSerial FFT")
#        123456789012345678901234567890123456789012345678901
println("Problem size (m) | Run time (sec.) | Gigaflops")
for e in 6:0.5:7
    m = round(Int, 10^e)
    @printf("%16d | ", m)
    t = runfft(m)
    @printf("%15.6f | %15.9f\n", t, 5e-9*m*log2(m)/t)
end
