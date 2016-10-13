using DistributedArrays

include("kernels_naive.jl") #For validation only
include("kernels_parallel.jl")

################################################################################

#Helper routines that initialize, execute, time, and validate the kernels

function runhplp(n)
    #Initialize
    hpl(distribute(rand(1, 1)), rand(1)) #Precompile
    A = drandn(n, n)
    b = randn(n)
    A′= Array(A)

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

function runrandomupdatep(m, nupdate=4m, seed=1)
    #Initialize
    randomupdate!(dzeros(UInt64,1), 1) #Precompile
    srand(seed)
    T = @DArray [i for i in 1:m]

    #Run
    t = @elapsed randomupdate!(T, nupdate)

    #Validate
    srand(seed)
    T′ = Array(T)
    randomupdate!(T′, nupdate)

    err = 0
    for i = 1:m
        err += (T′[i] != i)
    end
    err /= m
    if err > 0.01
        warn("Error rate = ", 100err, "%, exceeding 1% allowance")
    end
    return t
end

function runstreamtriadp(m, ntrials=10)
    #Initialize
    streamtriad!(distribute([0.0]), distribute([1.0]), 2.0, distribute([3.0]))
    a = dzeros(m)
    b = drandn(m)
    c = drandn(m)
    α = randn()

    @assert a.pids==workers() "Not enough work to parallelize over all cores"
    @assert a.pids==b.pids==c.pids "Cuts are not aligned"

    #Run
    t = Inf
    for i in 1:ntrials
        t0 = @elapsed streamtriad!(a, b, α, c)
        t = min(t, t0)
    end

    #Validate
    a′ = Array(a)
    d  = zeros(m)
    streamtriad!(d, Array(b), α, Array(c))
    err = norm(a′-d, 1)
    if err > 1e-6
        warn("Error = $err exceeds threshold")
    end
    return t
end

function runfftp(m)
    #Initialize
    FFT(distribute([1.0+0.0im]))
    z = @DArray [randn() + im*randn() for i in 1:m]
    z′ = Array(z)

    #Run
    t = @elapsed FFT(z)

    #Validate
    ifft!(Array(z))
    err = maxabs(z′ - z)
    errtol = 16*eps()*log2(m)
    if err > errtol
        warn("Error = $err exceeds tolerance of $errtol")
    end

    return t
end
