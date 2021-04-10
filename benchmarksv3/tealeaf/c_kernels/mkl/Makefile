vpath %.c $(KERNELS_PATH)

include $(KERNELS_PATH)/make.deps

KERNEL_SRCS := $(wildcard $(KERNELS_PATH)/*.c)
KERNEL_OBJS := $(patsubst $(KERNELS_PATH)/%.c, obj/$(KERNELS)/%.o, $(KERNEL_SRCS))

build_kernels: $(KERNEL_OBJS)
	@echo "Built MKL kernels"

CFLAGS += -DMKL_ILP64 -I${MKLROOT}/include

TL_LDFLAGS += $(OMP)

## MKL 11.3 OpenMP static
#TL_LDFLAGS += -Wl,--start-group \
#		  ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
#		  ${MKLROOT}/lib/intel64/libmkl_core.a \
#		  ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group \
#		  -lpthread -lm

## MKL 11.3 TBB Threading static
#TL_LDFLAGS += -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
#		  ${MKLROOT}/lib/intel64/libmkl_core.a \
#		  ${MKLROOT}/lib/intel64/libmkl_tbb_thread.a -Wl,--end-group \
#		  -ltbb -lstdc++ -lpthread -lm

##MKL 11.3.1 (OSX) ILP64 static
#TL_LDFLAGS += ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_core.a \
#		  ${MKLROOT}/lib/libmkl_intel_thread.a -lpthread -lm

## MKL 11.3
#TL_LDFLAGS += -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core \
#		  -lmkl_intel_thread -lmkl_blacs_intelmpi_lp64 -lpthread -lm

## MKL 11.2
#TL_LDFLAGS += -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 \
#		  -lmkl_core -lmkl_intel_thread -lmkl_blacs_intelmpi_lp64 -lpthread -lm

## MKL 11.2 Static linking
#TL_LDFLAGS += -Wl,--start-group \
#		  ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a \
#		  ${MKLROOT}/lib/intel64/libmkl_core.a \
#		  ${MKLROOT}/lib/intel64/libmkl_intel_thread.a \
#		  ${MKLROOT}/lib/intel64/libmkl_blacs_lp64.a \
#		  -Wl,--end-group -lpthread -lm

## MKL 11.3.1 (OSX)
#TL_LDFLAGS += -L${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_core \
#		  -lmkl_intel_thread -lpthread -lm
#

# MKL 11.3.1 (OSX) ILP64
TL_LDFLAGS +=  -L${MKLROOT}/lib -lmkl_intel_ilp64 -lmkl_core \
		   -lmkl_intel_thread -lpthread -lm

