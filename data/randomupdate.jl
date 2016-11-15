using Plots

data = readdlm(IOBuffer("""
    1 |           524288 |        1.026098 |     0.000510953
    2 |           524288 |        1.285015 |     0.000408001
    4 |           524288 |        1.286281 |     0.000407600
    8 |           524288 |        1.165687 |     0.000449767
   16 |           524288 |        1.103974 |     0.000474910
   32 |           524288 |        1.205526 |     0.000434904
   64 |           524288 |        1.112961 |     0.000471075
   """), '|')

data[:,4]*=524288/4096 #Correct reporting error

plot([1,64], [data[1,4], data[1,4]*64], c=:lightgrey, label="ideal")
plot!(data[:,1], data[:,4], xlabel="# cores", ylabel="Gflops", c=:blue, label="",
	xlims=(0,64), ylims=(0,0.1))
scatter!(data[:,1], data[:,4], c=:blue, label="")
png("../plots/randomupdate")
