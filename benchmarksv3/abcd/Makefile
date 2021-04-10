all: abcd.o Makefile
	nvcc $(DEFS) -arch=sm_70 -Xcompiler "-Wall -Wextra" abcd.o -o abcd

abcd.o: abcd.cu Makefile
	nvcc $(DEFS) -arch=sm_70 -Xcompiler "-Wall -Wextra" -ptx abcd.cu
	nvcc $(DEFS) -arch=sm_70 -Xcompiler "-Wall -Wextra" -c -Xptxas "-v" abcd.cu

clean:
	rm -f abcd.o abcd
