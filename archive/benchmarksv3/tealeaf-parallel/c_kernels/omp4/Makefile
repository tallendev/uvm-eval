vpath %.c $(KERNELS_PATH)

include $(KERNELS_PATH)/make.deps

KERNEL_SRCS := $(wildcard $(KERNELS_PATH)/*.c)
KERNEL_OBJS := $(patsubst $(KERNELS_PATH)/%.c, obj/$(KERNELS)/%.o, $(KERNEL_SRCS))

build_kernels: $(KERNEL_OBJS)
	@echo "Built OpenMP 4.0 kernels"
	-mv *.ptx *.cub obj/$(KERNELS)

CFLAGS += -DDIFFUSE_OVERLOAD
TL_LDFLAGS += -L/localhd/mattmar/lomp/lomp/source/lib64


