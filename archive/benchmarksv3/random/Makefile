all: page


page: page.cu
	nvcc $(DEFS) -Xptxas -v  -arch=sm_70  page.cu -o page
	#nvcc $(DEFS) -Xptxas -v  -Xptxas -O0 -Xcicc -O0 -arch=sm_70 -O0  page.cu -o page
	#nvcc -Xcompiler "-fopenmp" $(DEFS) -Xptxas -v  -Xptxas -O0 -Xcicc -O0 -arch=sm_70 -O0  page.cu -o page


clean:
	rm -f page


#nvcc $(DEFS) -G -g -Xptxas -v -arch=sm_70 -O0  band.cu -o band
