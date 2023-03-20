vpath %.cu $(KERNELS_PATH)

# Override the default compilation options
TL_LINK	  	 = $(CPP)
TL_COMPILER  = $(CPP)
TL_FLAGS 	 = $(CPPFLAGS) -g
CUDA_PATH = /usr/local/cuda
TL_LDFLAGS   += -L$(CUDA_PATH)/lib64 -lcudart $(OMP)

# TODO: Add more gencodes
NV_GENCODE = arch=compute_70,code=sm_70
KERNEL_FLAGS   = -I$(CUDA_PATH)/include -restrict -Xcompiler "-O3" \
			 -gencode $(NV_GENCODE)

KERNEL_COMPILER = nvcc

ifeq ($(DEBUG),yes)
  KERNEL_FLAGS += -O0 -g -G
else
  KERNEL_FLAGS += -O3 #--ptxas-options="-v"
endif

include $(KERNELS_PATH)/make.deps

KERNEL_SRCS := $(wildcard $(KERNELS_PATH)/*.cu)
KERNEL_OBJS := $(patsubst $(KERNELS_PATH)/%.cu, obj/$(KERNELS)/%.o, $(KERNEL_SRCS))

build_kernels: $(KERNEL_OBJS)
	@echo "Built CUDA kernels"

obj/$(KERNELS)/%.o: %.cu Makefile make.deps
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) -I$(KERNELS_PATH) $(OPTIONS) -c $< -o $@

