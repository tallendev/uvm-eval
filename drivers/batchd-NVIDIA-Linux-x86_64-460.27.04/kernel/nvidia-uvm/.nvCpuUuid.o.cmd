cmd_/home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-uvm/nvCpuUuid.o := cc -Wp,-MMD,/home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-uvm/.nvCpuUuid.o.d -nostdinc -isystem /usr/lib/gcc/x86_64-redhat-linux/10/include -I./arch/x86/include -I./arch/x86/include/generated  -I./include -I./arch/x86/include/uapi -I./arch/x86/include/generated/uapi -I./include/uapi -I./include/generated/uapi -include ./include/linux/kconfig.h -include ./include/linux/compiler_types.h -D__KERNEL__ -Wall -Wundef -Werror=strict-prototypes -Wno-trigraphs -fno-strict-aliasing -fno-common -fshort-wchar -fno-PIE -Werror=implicit-function-declaration -Werror=implicit-int -Wno-format-security -Wno-address-of-packed-member -std=gnu89 -mno-sse -mno-mmx -mno-sse2 -mno-3dnow -mno-avx -m64 -falign-jumps=1 -falign-loops=1 -mno-80387 -mno-fp-ret-in-387 -mpreferred-stack-boundary=3 -mskip-rax-setup -mtune=generic -mno-red-zone -mcmodel=kernel -Wno-sign-compare -fno-asynchronous-unwind-tables -mindirect-branch=thunk-extern -mindirect-branch-register -fno-jump-tables -fno-delete-null-pointer-checks -Wno-frame-address -Wno-format-truncation -Wno-format-overflow -Wno-address-of-packed-member -O2 -fno-allow-store-data-races -Wframe-larger-than=2048 -fstack-protector -Wno-unused-but-set-variable -Wimplicit-fallthrough -Wno-unused-const-variable -fno-var-tracking-assignments -g -pg -mrecord-mcount -mfentry -DCC_USING_FENTRY -Wdeclaration-after-statement -Wvla -Wno-pointer-sign -Wno-stringop-truncation -Wno-zero-length-bounds -Wno-array-bounds -Wno-stringop-overflow -Wno-restrict -Wno-maybe-uninitialized -fno-strict-overflow -fno-merge-all-constants -fmerge-constants -fno-stack-check -fconserve-stack -Werror=date-time -Werror=incompatible-pointer-types -Werror=designated-init -fmacro-prefix-map=./= -fcf-protection=none -Wno-packed-not-aligned -I/home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel/common/inc -I/home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel -Wall -MD -Wno-cast-qual -Wno-error -Wno-format-extra-args -D__KERNEL__ -DMODULE -DNVRM -DNV_VERSION_STRING=\"460.27.04\" -Wno-unused-function -Wuninitialized -fno-strict-aliasing -mno-red-zone -mcmodel=kernel -DNV_UVM_ENABLE -Werror=undef -DNV_SPECTRE_V2=0 -DNV_KERNEL_INTERFACE_LAYER -O2 -DNVIDIA_UVM_ENABLED -DNVIDIA_UNDEF_LEGACY_BIT_MACROS -DLinux -D__linux__ -I/home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-uvm  -DMODULE  -DKBUILD_BASENAME='"nvCpuUuid"' -DKBUILD_MODNAME='"nvidia_uvm"' -c -o /home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-uvm/nvCpuUuid.o /home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-uvm/nvCpuUuid.c

source_/home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-uvm/nvCpuUuid.o := /home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-uvm/nvCpuUuid.c

deps_/home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-uvm/nvCpuUuid.o := \
  include/linux/kconfig.h \
    $(wildcard include/config/cc/version/text.h) \
    $(wildcard include/config/cpu/big/endian.h) \
    $(wildcard include/config/booger.h) \
    $(wildcard include/config/foo.h) \
  include/linux/compiler_types.h \
    $(wildcard include/config/have/arch/compiler/h.h) \
    $(wildcard include/config/enable/must/check.h) \
    $(wildcard include/config/cc/has/asm/inline.h) \
  include/linux/compiler_attributes.h \
  include/linux/compiler-gcc.h \
    $(wildcard include/config/retpoline.h) \
    $(wildcard include/config/arch/use/builtin/bswap.h) \
  /home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel/common/inc/nvtypes.h \
  /home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel/common/inc/cpuopsys.h \
  /home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel/common/inc/xapi-sdk.h \
  /home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel/common/inc/nvCpuUuid.h \

/home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-uvm/nvCpuUuid.o: $(deps_/home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-uvm/nvCpuUuid.o)

$(deps_/home/tnallen/cuda11.2/batchd-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-uvm/nvCpuUuid.o):
