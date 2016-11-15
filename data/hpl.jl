using Plots

data = readdlm(IOBuffer("""
    1 |             2000 |       16.234234 |     0.328893463
    2 |             2000 |       17.744895 |     0.300894045
    4 |             2000 |        7.110266 |     0.750933013
    8 |             2000 |        4.138830 |     1.290058600
   16 |             2000 |        3.829481 |     1.394270721
   32 |             2000 |        3.991391 |     1.337712400
   64 |             2000 |        4.183819 |     1.276186478
"""), '|')

plot!([1,64], [data[1,4], data[1,4]*64], c=:grey, label="ideal Julia")
plot!(data[:,1], data[:,4], xlabel="# cores", ylabel="Gflops", c=:blue, label="Julia", ylims=(0, 1.5), xlims=(0, 80))
scatter!(data[:,1], data[:,4], c=:blue, label="")
png("../plots/hpl")
