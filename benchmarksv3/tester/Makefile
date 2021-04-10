all: tester.o
	nvcc $(DEFS) -Xcompiler "-Wall -Wextra" tester.o -o tester

tester.o: tester.cu
	nvcc $(DEFS) -Xcompiler "-Wall -Wextra" -c tester.cu

clean:
	rm -f tester.o tester
