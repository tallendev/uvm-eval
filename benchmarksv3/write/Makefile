#NVFLAGS=-gencode arch=compute_70,code=sm_70
NVFLAGS=-arch=sm_70

all: write.o
	nvcc $(DEFS) $(NVFLAGS)  -Xcompiler "-Wall -Wextra" write.o -o write

write.o: write.cu
	nvcc $(DEFS) $(NVFLAGS) -Xcompiler "-Wall -Wextra" -c write.cu

clean:
	rm -f write.o write
