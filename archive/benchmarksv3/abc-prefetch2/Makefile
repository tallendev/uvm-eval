all: abc.o
	nvcc $(DEFS) -arch=sm_70 -Xcompiler "-Wall -Wextra" abc.o -o abc

abc.o: abc.cu
	nvcc $(DEFS) -arch=sm_70 -Xcompiler "-Wall -Wextra" -c abc.cu

clean:
	rm -f abc.o abc
