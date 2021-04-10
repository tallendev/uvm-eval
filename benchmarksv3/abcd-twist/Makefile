all: abcd-twist.o Makefile
	nvcc $(DEFS) -arch=sm_70 -Xcompiler "-Wall -Wextra" abcd-twist.o -o abcd-twist

abcd-twist.o: abcd-twist.cu Makefile
	nvcc $(DEFS) -arch=sm_70 -Xcompiler "-Wall -Wextra" -ptx abcd-twist.cu
	nvcc $(DEFS) -arch=sm_70 -Xcompiler "-Wall -Wextra" -c -Xptxas "-v" abcd-twist.cu

clean:
	rm -f abcd-twist.o abcd-twist
