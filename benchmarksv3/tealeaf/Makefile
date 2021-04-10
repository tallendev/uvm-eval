# Used specified parameters
KERNELS   = cuda
CPROFILER = yes
DEBUG     = no
OPTIONS   = #-DNO_MPI
CC        = mpicc
CPP       = mpic++
COMPILER  = CUDA

# By default use C compiler and flags, may be overridden by implementation
TL_LINK     = $(CC)
TL_LDFLAGS  = -lrt -lm
TL_COMPILER = $(CC)
TL_FLAGS    = $(CFLAGS) -g

include make.flags

# Apply additional options
ifeq ($(CPROFILER), yes)
  OPTIONS += -DENABLE_PROFILING
endif

KERNELS_PATH = c_kernels/$(KERNELS)

vpath %.c drivers/

# Get all control and driver objects
OBJS  = $(patsubst %.c, obj/$(KERNELS)/%.o, $(wildcard *.c))
OBJS += $(patsubst drivers/%.c, obj/$(KERNELS)/%.o, $(wildcard drivers/*.c))

# Link together control code, drivers and kernels
tealeaf: make_build_dir build_kernels $(OBJS) Makefile
	$(TL_LINK) $(TL_FLAGS) $(OBJS) $(KERNEL_OBJS) $(TL_LDFLAGS) -o tealeaf.$(KERNELS)
	cp tealeaf.$(KERNELS) tealeaf

include make.deps
include $(KERNELS_PATH)/Makefile

# Rule to make controlling code
obj/$(KERNELS)/%.o: %.c Makefile make.deps $(KERNELS_PATH)/make.deps
	$(TL_COMPILER) $(TL_FLAGS) -I/usr/local/cuda/include -I$(KERNELS_PATH) $(OPTIONS) -c $< -o $@

# Create the object file directory and implementation sub-directory
make_build_dir:
	@mkdir -p obj/
	@mkdir -p obj/$(KERNELS)

.PHONY: clean

clean:
	rm -rf obj/ *.o *.a tealeaf

## Populates make.deps with dependencies
#obj/%.o: %.cu Makefile make.deps
#	gcc -MM $< >> new.make.deps

