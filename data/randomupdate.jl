using Plots

data = readdlm(IOBuffer("""
    1 |             4096 |        0.085288 |     0.000048026
    2 |             4096 |        0.085923 |     0.000047670
    4 |             4096 |        0.103074 |     0.000039738
    8 |             4096 |        0.088835 |     0.000046108
   16 |             4096 |        0.103771 |     0.000039471
   32 |             4096 |        0.094921 |     0.000043151
   64 |             4096 |        0.102273 |     0.000040050
   """), '|')

plot([1,64], [data[1,4], data[1,4]*64], c=:lightgrey, label="ideal")
plot!(data[:,1], data[:,4], xlabel="# cores", ylabel="Gflops", c=:blue, label="")
scatter!(data[:,1], data[:,4], c=:blue, label="")
png("../plots/randomupdate")
