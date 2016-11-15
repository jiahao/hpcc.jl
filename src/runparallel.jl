#Run the tests

include("helper_parallel.jl")

println("Parallel HPL: Tuning step on one kernel")
#        1234567890123456789012345678901234567890123456
println("    k | Problem size (n) | Run time (sec.) | Gigaflops")

ks = collect(128*(1:5))
topt = Inf
kopt = 0
for k in ks
    @printf("%5d | ", k)
    n = 1000
    @printf("%16d | ", n)
    t = runhplp(n, k, k)
    if t < topt
        topt = t
        kopt = k
    end
    @printf("%15.6f | %15.9f\n", t, 1e-9*(2n^3/3 + 3n^2/2)/t)
end

info("Setting block size to k = $kopt for HPL")

for np in [1,2,4,8,16,32,64]#1:64#Sys.CPU_CORES
    ps = addprocs(min(np-nworkers(), 1))

#Broadcast function definitions to all workers
z = [@spawnat p include("helper_parallel.jl") for p in ps]
[fetch(w) for w in z]

println("Parallel HPL")
#        1234567890123456789012345678901234567890123456
println("Cores | Problem size (n) | Run time (sec.) | Gigaflops")
for e in 3:0.2:3.8
    @printf("%5d | ", nworkers())
    n = round(Int, 10^e)
    @printf("%16d | ", n)
    t = runhplp(n, kopt, kopt)
    @printf("%15.6f | %15.9f\n", t, 1e-9*(2n^3/3 + 3n^2/2)/t)
end

println("\n\n\nParallel random update")
#        12345678901234567890123456789012345678901234567890123
println("Cores | Problem size (m) | Run time (sec.) | Gigaupdates/sec.")
for e in 10:12#18:24
    @printf("%5d | ", nworkers())
    m = 2^e
    @printf("%16d | ", m)
    t = runrandomupdatep(m)
    @printf("%15.6f | %15.9f\n", t, m*1e-9/t)
end

println("\n\n\nParallel STREAM triad")
#        123456789012345678901234567890123456789012345678901
println("Cores | Problem size (m) | Run time (sec.) | Gigabytes/sec.")
for e in 6.5:0.5:8.0
    @printf("%5d | ", nworkers())
    m = round(Int, 10^e)
    @printf("%16d | ", m)
    t = runstreamtriadp(m)
    @printf("%15.6f | %15.9f\n", t, 1e-9*24*m/t)
end

#Not implemented
#println("\n\n\nParallel FFT")
##        123456789012345678901234567890123456789012345678901
#println("Cores | Problem size (m) | Run time (sec.) | Gigaflops")
#for e in 6:0.5:7
#    @printf("%5d | ", nworkers())
#    m = round(Int, 10^e)
#    @printf("%16d | ", m)
#    t = runfftp(m)
#    @printf("%15.6f | %15.9f\n", t, 5e-9*m*log2(m)/t)
#end

end #addprocs
