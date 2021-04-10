#NVFLAGS=-gencode arch=compute_70,code=sm_70
NVFLAGS=-arch=sm_70

all: abc.o
	nvcc $(DEFS) $(NVFLAGS)  -Xcompiler "-Wall -Wextra" abc.o -o abc

abc.o: abc.cu
	nvcc $(DEFS) $(NVFLAGS) -Xcompiler "-Wall -Wextra" -c abc.cu

clean:
	rm -f abc.o abc
