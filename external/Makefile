arch=$(shell uname -s)
numcores=$(shell julia -e 'print(Sys.CPU_CORES)')

ifeq ($(arch), Darwin)
	arch=MacOSX
endif

.PHONY: distclean clean results

results: output/randomaccess.txt output/triad.txt output/hpl.txt

clean:
	rm -rf output
	
distclean: clean
	rm -rf hpcc-code

output/randomaccess.txt: output/$(numcores).txt
	grep MPIRandomAccess_GUPs output/*.txt | sed 's/output\///' | sed 's/\.txt:MPIRandomAccess_GUPs=/ /' | sort -n > $@

output/triad.txt: output/$(numcores).txt
	grep StarSTREAM_Triad output/*.txt | sed 's/output\///' | sed 's/\.txt:StarSTREAM_Triad=/ /' | sort -n > $@

output/hpl.txt: output/$(numcores).txt
	grep HPL_Tflops output/*.txt | sed 's/output\///' | sed 's/\.txt:HPL_Tflops=/ /' | sort -n > $@

output/$(numcores).txt: hpcc-code/hpcc hpcc-code/hpccinf.txt
	mkdir -p output
	rm -f hpcc-code/hpccoutf.txt
	cd hpcc-code && for np in `seq 1 $(numcores)`; do mpirun -np $$np ./hpcc && mv hpccoutf.txt ../output/$$np.txt; done

hpcc-code/hpcc: hpcc-code/hpl/Make.$(arch)
	cd hpcc-code && make arch=$(arch)

hpcc-code/hpl/Make.Linux: hpcc-code
	sed 's/LAlib        = -framework Accelerate/LAlib        = -llapack -lblas/' hpcc-code/hpl/Make.MacOSX > $@

hpcc-code/hpccinf.txt: hpcc-code
	#Use default setup file
	cp hpcc-code/hpl/testing/ptest/HPL.dat $@

hpcc-code:
	hg clone http://hg.code.sf.net/p/hpcc/code hpcc-code
