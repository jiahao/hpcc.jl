#Simple parallel reduction for random eigenvalue computation

N =1024
t1=100
include("randeig.jl")
nps = Float64[]
gflops = Float64[]
for np in [1,2,4,8,16,32,64]#1:64#Sys.CPU_CORES
    ps = addprocs(np==1 ? 1 : np-nworkers())
    @sync for p in ps
        @spawnat p include("randeig.jl")
    end
    t = @elapsed randeig(t1*np, N)
    gflop = t1*np*4/3*N^3*1e-9/t
    println(nworkers(), '\t', t, '\t', gflop)
    push!(nps, nworkers())
    push!(gflops, gflop)
end

using Plots
plot(nps, gflops, c = :blue, label = "")
scatter!(nps, gflops, c = :blue, label="")
plot!([extrema(nps)...], [gflops[1], nps[end]*gflops[1], c = :grey, label="Ideal"]) 
png("../plots/parfor")

