#Run the tests

include("helper_parallel.jl")

for np in 1:Sys.CPU_CORES
ps = addprocs(1)

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
    t = runhplp(n)
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
    @printf("%15.6f | %15.9f\n", t, 1e-9*24*10*m/t)
end

println("\n\n\nParallel FFT")
#        123456789012345678901234567890123456789012345678901
println("Cores | Problem size (m) | Run time (sec.) | Gigaflops")
for e in 6:0.5:7
    @printf("%5d | ", nworkers())
    m = round(Int, 10^e)
    @printf("%16d | ", m)
    t = runfftp(m)
    @printf("%15.6f | %15.9f\n", t, 5e-9*m*log2(m)/t)
end

end #addprocs
