using Plots

data = readdlm(IOBuffer("""
    1 |        100000000 |        1.069432 |    2.2441811441
    2 |        100000000 |        0.573543 |    4.1845140477
    4 |        100000000 |        0.381410 |    6.2924386313
    8 |        100000000 |        0.133315 |   18.0024743951
   16 |        100000000 |        0.087866 |   27.3143817048
   32 |        100000000 |        0.094582 |   25.3748789460
   64 |        100000000 |        0.082351 |   29.1436764549
   """), '|')

mpidata = readdlm(IOBuffer("""
1.          4791.9     0.506998     0.500844     0.515415
2.          9366.9     0.256951     0.256222     0.257966
4.         18607.7     0.153044     0.128979     0.243534
8.         37209.9     0.095361     0.064499     0.206526
16         43776.9     0.081599     0.054823     0.129850
32         65392.5     0.061542     0.036701     0.082718
64         81010.2     0.056582     0.029626     0.081922
"""))

mpidata[:,2] ./= 1024 #Convert from MB/s to GB/s

plot([1,64], [mpidata[1,2], mpidata[1,2]*64], c=:lightgrey, label="ideal MPI")
plot!(mpidata[:,1], mpidata[:,2], c=:red, label="C/MPI")
scatter!(mpidata[:,1], mpidata[:,2], c=:red, label="")
plot!([1,64], [data[1,4], data[1,4]*64], c=:grey, label="ideal Julia")
plot!(data[:,1], data[:,4], xlabel="# cores", ylabel="Gflops", c=:blue, label="Julia", ylims=(0, 90), xlims=(0, 80))
scatter!(data[:,1], data[:,4], c=:blue, label="")
png("../plots/triad")
