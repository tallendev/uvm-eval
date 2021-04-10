cmd_/home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-drm/nvidia-drm-format.o := cc -Wp,-MMD,/home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-drm/.nvidia-drm-format.o.d -nostdinc -isystem /usr/lib/gcc/x86_64-redhat-linux/10/include -I./arch/x86/include -I./arch/x86/include/generated  -I./include -I./arch/x86/include/uapi -I./arch/x86/include/generated/uapi -I./include/uapi -I./include/generated/uapi -include ./include/linux/kconfig.h -include ./include/linux/compiler_types.h -D__KERNEL__ -Wall -Wundef -Werror=strict-prototypes -Wno-trigraphs -fno-strict-aliasing -fno-common -fshort-wchar -fno-PIE -Werror=implicit-function-declaration -Werror=implicit-int -Wno-format-security -Wno-address-of-packed-member -std=gnu89 -mno-sse -mno-mmx -mno-sse2 -mno-3dnow -mno-avx -m64 -falign-jumps=1 -falign-loops=1 -mno-80387 -mno-fp-ret-in-387 -mpreferred-stack-boundary=3 -mskip-rax-setup -mtune=generic -mno-red-zone -mcmodel=kernel -Wno-sign-compare -fno-asynchronous-unwind-tables -mindirect-branch=thunk-extern -mindirect-branch-register -fno-jump-tables -fno-delete-null-pointer-checks -Wno-frame-address -Wno-format-truncation -Wno-format-overflow -Wno-address-of-packed-member -O2 -fno-allow-store-data-races -Wframe-larger-than=2048 -fstack-protector -Wno-unused-but-set-variable -Wimplicit-fallthrough -Wno-unused-const-variable -fno-var-tracking-assignments -g -pg -mrecord-mcount -mfentry -DCC_USING_FENTRY -Wdeclaration-after-statement -Wvla -Wno-pointer-sign -Wno-stringop-truncation -Wno-zero-length-bounds -Wno-array-bounds -Wno-stringop-overflow -Wno-restrict -Wno-maybe-uninitialized -fno-strict-overflow -fno-merge-all-constants -fmerge-constants -fno-stack-check -fconserve-stack -Werror=date-time -Werror=incompatible-pointer-types -Werror=designated-init -fmacro-prefix-map=./= -fcf-protection=none -Wno-packed-not-aligned -I/home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/common/inc -I/home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel -Wall -MD -Wno-cast-qual -Wno-error -Wno-format-extra-args -D__KERNEL__ -DMODULE -DNVRM -DNV_VERSION_STRING=\"460.27.04\" -Wno-unused-function -Wuninitialized -fno-strict-aliasing -mno-red-zone -mcmodel=kernel -DNV_UVM_ENABLE -Werror=undef -DNV_SPECTRE_V2=0 -DNV_KERNEL_INTERFACE_LAYER -I/home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-drm -UDEBUG -U_DEBUG -DNDEBUG -DNV_BUILD_MODULE_INSTANCES=0  -DMODULE  -DKBUILD_BASENAME='"nvidia_drm_format"' -DKBUILD_MODNAME='"nvidia_drm"' -c -o /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-drm/nvidia-drm-format.o /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-drm/nvidia-drm-format.c

source_/home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-drm/nvidia-drm-format.o := /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-drm/nvidia-drm-format.c

deps_/home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-drm/nvidia-drm-format.o := \
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
  /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-drm/nvidia-drm-conftest.h \
  /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/common/inc/conftest.h \
  /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/conftest/headers.h \
  /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/conftest/functions.h \
  /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/conftest/generic.h \
  /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/conftest/macros.h \
  /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/conftest/symbols.h \
  /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/conftest/types.h \
  include/linux/kernel.h \
    $(wildcard include/config/preempt/voluntary.h) \
    $(wildcard include/config/debug/atomic/sleep.h) \
    $(wildcard include/config/preempt/rt.h) \
    $(wildcard include/config/mmu.h) \
    $(wildcard include/config/prove/locking.h) \
    $(wildcard include/config/smp.h) \
    $(wildcard include/config/panic/timeout.h) \
    $(wildcard include/config/tracing.h) \
    $(wildcard include/config/ftrace/mcount/record.h) \
  include/linux/limits.h \
  include/uapi/linux/limits.h \
  include/linux/types.h \
    $(wildcard include/config/have/uid16.h) \
    $(wildcard include/config/uid16.h) \
    $(wildcard include/config/arch/dma/addr/t/64bit.h) \
    $(wildcard include/config/phys/addr/t/64bit.h) \
    $(wildcard include/config/64bit.h) \
  include/uapi/linux/types.h \
  arch/x86/include/generated/uapi/asm/types.h \
  include/uapi/asm-generic/types.h \
  include/asm-generic/int-ll64.h \
  include/uapi/asm-generic/int-ll64.h \
  arch/x86/include/uapi/asm/bitsperlong.h \
  include/asm-generic/bitsperlong.h \
  include/uapi/asm-generic/bitsperlong.h \
  include/uapi/linux/posix_types.h \
  include/linux/stddef.h \
  include/uapi/linux/stddef.h \
  include/linux/compiler_types.h \
  arch/x86/include/asm/posix_types.h \
    $(wildcard include/config/x86/32.h) \
  arch/x86/include/uapi/asm/posix_types_64.h \
  include/uapi/asm-generic/posix_types.h \
  include/vdso/limits.h \
  include/linux/linkage.h \
    $(wildcard include/config/arch/use/sym/annotations.h) \
  include/linux/stringify.h \
  include/linux/export.h \
    $(wildcard include/config/modversions.h) \
    $(wildcard include/config/module/rel/crcs.h) \
    $(wildcard include/config/have/arch/prel32/relocations.h) \
    $(wildcard include/config/modules.h) \
    $(wildcard include/config/trim/unused/ksyms.h) \
    $(wildcard include/config/unused/symbols.h) \
  include/linux/compiler.h \
    $(wildcard include/config/trace/branch/profiling.h) \
    $(wildcard include/config/profile/all/branches.h) \
    $(wildcard include/config/stack/validation.h) \
  arch/x86/include/generated/asm/rwonce.h \
  include/asm-generic/rwonce.h \
  include/linux/kasan-checks.h \
    $(wildcard include/config/kasan.h) \
  include/linux/kcsan-checks.h \
    $(wildcard include/config/kcsan.h) \
    $(wildcard include/config/kcsan/ignore/atomics.h) \
  arch/x86/include/asm/linkage.h \
    $(wildcard include/config/x86/64.h) \
    $(wildcard include/config/x86/alignment/16.h) \
  include/linux/bitops.h \
  include/linux/bits.h \
  include/linux/const.h \
  include/vdso/const.h \
  include/uapi/linux/const.h \
  include/vdso/bits.h \
  include/linux/build_bug.h \
  arch/x86/include/asm/bitops.h \
    $(wildcard include/config/x86/cmov.h) \
  arch/x86/include/asm/alternative.h \
  arch/x86/include/asm/asm.h \
  arch/x86/include/asm/rmwcc.h \
    $(wildcard include/config/cc/has/asm/goto.h) \
  arch/x86/include/asm/barrier.h \
  arch/x86/include/asm/nops.h \
    $(wildcard include/config/mk7.h) \
    $(wildcard include/config/x86/p6/nop.h) \
  include/asm-generic/barrier.h \
  include/asm-generic/bitops/find.h \
    $(wildcard include/config/generic/find/first/bit.h) \
  include/asm-generic/bitops/sched.h \
  arch/x86/include/asm/arch_hweight.h \
  arch/x86/include/asm/cpufeatures.h \
  arch/x86/include/asm/required-features.h \
    $(wildcard include/config/x86/minimum/cpu/family.h) \
    $(wildcard include/config/math/emulation.h) \
    $(wildcard include/config/x86/pae.h) \
    $(wildcard include/config/x86/cmpxchg64.h) \
    $(wildcard include/config/x86/use/3dnow.h) \
    $(wildcard include/config/matom.h) \
    $(wildcard include/config/paravirt.h) \
  arch/x86/include/asm/disabled-features.h \
    $(wildcard include/config/x86/smap.h) \
    $(wildcard include/config/x86/umip.h) \
    $(wildcard include/config/x86/intel/memory/protection/keys.h) \
    $(wildcard include/config/x86/5level.h) \
    $(wildcard include/config/page/table/isolation.h) \
  include/asm-generic/bitops/const_hweight.h \
  include/asm-generic/bitops/instrumented-atomic.h \
  include/linux/instrumented.h \
  include/asm-generic/bitops/instrumented-non-atomic.h \
  include/asm-generic/bitops/instrumented-lock.h \
  include/asm-generic/bitops/le.h \
  arch/x86/include/uapi/asm/byteorder.h \
  include/linux/byteorder/little_endian.h \
  include/uapi/linux/byteorder/little_endian.h \
  include/linux/swab.h \
  include/uapi/linux/swab.h \
  arch/x86/include/uapi/asm/swab.h \
  include/linux/byteorder/generic.h \
  include/asm-generic/bitops/ext2-atomic-setbit.h \
  include/linux/log2.h \
    $(wildcard include/config/arch/has/ilog2/u32.h) \
    $(wildcard include/config/arch/has/ilog2/u64.h) \
  include/linux/typecheck.h \
  include/linux/printk.h \
    $(wildcard include/config/message/loglevel/default.h) \
    $(wildcard include/config/console/loglevel/default.h) \
    $(wildcard include/config/console/loglevel/quiet.h) \
    $(wildcard include/config/early/printk.h) \
    $(wildcard include/config/printk/nmi.h) \
    $(wildcard include/config/printk.h) \
    $(wildcard include/config/dynamic/debug.h) \
    $(wildcard include/config/dynamic/debug/core.h) \
  include/linux/init.h \
    $(wildcard include/config/strict/kernel/rwx.h) \
    $(wildcard include/config/strict/module/rwx.h) \
  include/linux/kern_levels.h \
  include/linux/cache.h \
    $(wildcard include/config/arch/has/cache/line/size.h) \
  include/uapi/linux/kernel.h \
  include/uapi/linux/sysinfo.h \
  arch/x86/include/asm/cache.h \
    $(wildcard include/config/x86/l1/cache/shift.h) \
    $(wildcard include/config/x86/internode/cache/shift.h) \
    $(wildcard include/config/x86/vsmp.h) \
  include/linux/ratelimit_types.h \
  include/uapi/linux/param.h \
  arch/x86/include/generated/uapi/asm/param.h \
  include/asm-generic/param.h \
    $(wildcard include/config/hz.h) \
  include/uapi/asm-generic/param.h \
  include/linux/spinlock_types.h \
    $(wildcard include/config/debug/spinlock.h) \
    $(wildcard include/config/debug/lock/alloc.h) \
  arch/x86/include/asm/spinlock_types.h \
  include/asm-generic/qspinlock_types.h \
    $(wildcard include/config/nr/cpus.h) \
  include/asm-generic/qrwlock_types.h \
  include/linux/lockdep_types.h \
    $(wildcard include/config/prove/raw/lock/nesting.h) \
    $(wildcard include/config/preempt/lock.h) \
    $(wildcard include/config/lockdep.h) \
    $(wildcard include/config/lock/stat.h) \
  include/linux/rwlock_types.h \
  include/linux/dynamic_debug.h \
    $(wildcard include/config/jump/label.h) \
  include/linux/jump_label.h \
    $(wildcard include/config/have/arch/jump/label/relative.h) \
  arch/x86/include/asm/jump_label.h \
  arch/x86/include/asm/div64.h \
  include/asm-generic/div64.h \
  /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-drm/nvidia-drm-format.h \
  include/drm/drm_fourcc.h \
  include/uapi/drm/drm_fourcc.h \
  include/uapi/drm/drm.h \
  arch/x86/include/generated/uapi/asm/ioctl.h \
  include/asm-generic/ioctl.h \
  include/uapi/asm-generic/ioctl.h \
  include/uapi/drm/drm_mode.h \
  /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/common/inc/nvkms-format.h \
  /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/common/inc/nvtypes.h \
  /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/common/inc/cpuopsys.h \
  /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/common/inc/xapi-sdk.h \
  /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-drm/nvidia-drm-os-interface.h \
  /home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/common/inc/nvtypes.h \

/home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-drm/nvidia-drm-format.o: $(deps_/home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-drm/nvidia-drm-format.o)

$(deps_/home/tnallen/cuda11.2/faults-NVIDIA-Linux-x86_64-460.27.04/kernel/nvidia-drm/nvidia-drm-format.o):
