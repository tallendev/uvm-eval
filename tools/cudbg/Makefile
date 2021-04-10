all: tester.o
	nvcc -G -g -Xcompiler "-Wall -Wextra" tester.o -o tester

tester.o: tester.cu
	nvcc -G -g -Xcompiler "-Wall -Wextra" -c tester.cu

clean:
	rm -f tester.o tester
