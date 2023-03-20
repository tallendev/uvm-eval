vpath %.c $(KERNELS_PATH)

include $(KERNELS_PATH)/make.deps

KERNEL_SRCS := $(wildcard $(KERNELS_PATH)/*.c)
KERNEL_OBJS := $(patsubst $(KERNELS_PATH)/%.c, obj/$(KERNELS)/%.o, $(KERNEL_SRCS))

build_kernels: $(KERNEL_OBJS)
	@echo "Built OpenACC kernels"
	-mv *.lst *.ptx *.cub obj/$(KERNELS)/

TL_LDFLAGS	+= -acc

ifeq ($(OPENACC_TARGET), HSW)
    OACC_FLAGS = -ta=multicore -tp=haswell-64 
endif
ifeq ($(OPENACC_TARGET), IVB)
    OACC_FLAGS += -ta=multicore -tp=ivybridge-64 
endif
ifeq ($(OPENACC_TARGET), GPU)
    OACC_FLAGS += -ta=tesla,cc35,kepler -tp=ivybridge-64
endif
ifndef OPENACC_TARGET
    OACC_FLAGS += -ta=tesla,cc35,kepler -tp=ivybridge-64
endif

TL_LDFLAGS 	+= -acc $(OACC_FLAGS)
CFLAGS 		+= -acc $(OACC_FLAGS) -Minfo=accel -DDIFFUSE_OVERLOAD

