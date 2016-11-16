arch=$(shell uname -s)
numcores=$(shell julia -e 'print(Sys.CPU_CORES)')

ifeq ($(arch), Darwin)
	arch=MacOSX
endif

output/$(numcores).txt: hpcc-code/hpcc hpcc-code/hpccinf.txt
	mkdir -p output
	cd hpcc-code && for np in `seq 1 $(numcores)`; do mpirun -np $$np ./hpcc && mv hpcc-code/hpccoutf.txt output/$$np.txt; done

hpcc-code/hpcc: hpcc-code/hpl/Make.$(arch)
	cd hpcc-code && make arch=$(arch)

hpcc-code/hpl/Make.Linux: hpcc-code
	sed 's/LAlib        = -framework Accelerate/LAlib        = -llapack -lblas/' hpcc-code/hpl/Make.MacOSX > $@

hpcc-code/hpccinf.txt: hpcc-code
	#Use default setup file
	cp hpcc-code/hpl/testing/ptest/HPL.dat $@

hpcc-code:
	hg clone http://hg.code.sf.net/p/hpcc/code hpcc-code
