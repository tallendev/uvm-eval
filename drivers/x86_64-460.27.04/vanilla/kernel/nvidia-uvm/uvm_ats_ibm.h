/*******************************************************************************
    Copyright (c) 2018-2019 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#ifndef __UVM_ATS_IBM_H__
#define __UVM_ATS_IBM_H__

#include "uvm_linux.h"
#include "uvm_forward_decl.h"
#include "uvm_hal_types.h"

#if defined(NVCPU_PPC64LE) && defined(NV_PNV_PCI_GET_NPU_DEV_PRESENT)
    #include <asm/mmu.h>
    #if defined(NV_MAX_NPUS)
        #define UVM_IBM_NPU_SUPPORTED() 1
    #else
        #define UVM_IBM_NPU_SUPPORTED() 0
    #endif
#else
    #define UVM_IBM_NPU_SUPPORTED() 0
#endif

#if defined(NV_ASM_OPAL_API_H_PRESENT)
    // For OPAL_NPU_INIT_CONTEXT
    #include <asm/opal-api.h>
#endif

// Timeline of kernel changes:
//
// 0) Before 1ab66d1fbadad86b1f4a9c7857e193af0ee0022c
//      - No NPU-ATS code existed, nor did the OPAL_NPU_INIT_CONTEXT firmware
//        call.
//      - NV_PNV_NPU2_INIT_CONTEXT_PRESENT                  Not defined
//      - NV_PNV_NPU2_INIT_CONTEXT_CALLBACK_RETURNS_VOID    Not defined
//      - OPAL_NPU_INIT_CONTEXT                             Not defined
//      - ATS support type                                  None
//
// 1) NPU ATS code added: 1ab66d1fbadad86b1f4a9c7857e193af0ee0022c, v4.12
//    (2017-04-03)
//      - This commit added initial support for NPU ATS, including the necessary
//        OPAL firmware calls. This support was developmental and required
//        several bug fixes before it could be used in production.
//      - NV_PNV_NPU2_INIT_CONTEXT_PRESENT                  Defined
//      - NV_PNV_NPU2_INIT_CONTEXT_CALLBACK_RETURNS_VOID    Not defined
//      - OPAL_NPU_INIT_CONTEXT                             Defined
//      - ATS support type                                  None
//
// 2) NPU ATS code fixed: a1409adac748f0db655e096521bbe6904aadeb98, v4.17
//    (2018-04-11)
//      - This commit changed the function signature for pnv_npu2_init_context's
//        callback parameter. Since all required bug fixes went in prior to this
//        change, we can use the callback signature as a flag to indicate
//        whether the PPC arch layer in the kernel supports ATS in production.
//      - NV_PNV_NPU2_INIT_CONTEXT_PRESENT                  Defined
//      - NV_PNV_NPU2_INIT_CONTEXT_CALLBACK_RETURNS_VOID    Defined
//      - OPAL_NPU_INIT_CONTEXT                             Defined
//      - ATS support type                                  Kernel
//
// 3) NPU ATS code removed: 7eb3cf761927b2687164e182efa675e6c09cfe44, v5.3
//    (2019-06-25)
//      - This commit removed NPU-ATS support from the PPC arch layer, so the
//        driver needs to handle things instead. pnv_npu2_init_context is no
//        longer present, so we use OPAL_NPU_INIT_CONTEXT to differentiate
//        between this state and scenario #0.
//      - NV_PNV_NPU2_INIT_CONTEXT_PRESENT                  Not defined
//      - NV_PNV_NPU2_INIT_CONTEXT_CALLBACK_RETURNS_VOID    Not defined
//      - OPAL_NPU_INIT_CONTEXT                             Defined
//      - ATS support type                                  Driver
//
#if defined(NV_PNV_NPU2_INIT_CONTEXT_CALLBACK_RETURNS_VOID)
    #define UVM_ATS_IBM_SUPPORTED_IN_KERNEL()   1
    #define UVM_ATS_IBM_SUPPORTED_IN_DRIVER()   0
#elif !defined(NV_PNV_NPU2_INIT_CONTEXT_PRESENT) && defined(OPAL_NPU_INIT_CONTEXT) && UVM_CAN_USE_MMU_NOTIFIERS()
    #define UVM_ATS_IBM_SUPPORTED_IN_KERNEL()   0
    #define UVM_ATS_IBM_SUPPORTED_IN_DRIVER()   1
#else
    #define UVM_ATS_IBM_SUPPORTED_IN_KERNEL()   0
    #define UVM_ATS_IBM_SUPPORTED_IN_DRIVER()   0
#endif

#define UVM_ATS_IBM_SUPPORTED() (UVM_ATS_IBM_SUPPORTED_IN_KERNEL() || UVM_ATS_IBM_SUPPORTED_IN_DRIVER())

// Maximum number of parallel ATSD register sets per NPU
#define UVM_MAX_ATSD_REGS 16

typedef struct
{
    // Number of retained GPUs under this NPU. The other fields in this struct
    // are only valid if this is non-zero.
    unsigned int num_retained_gpus;

    // PCI domain containing this NPU. This acts as a unique system-wide ID for
    // this UVM NPU.
    int pci_domain;

    // The ATS-related fields are only valid when ATS support is enabled and
    // UVM_ATS_IBM_SUPPORTED_IN_DRIVER() is 1.
    struct
    {
        // Mapped addresses of the ATSD trigger registers. There may be more
        // than one set of identical registers per NPU to enable concurrent
        // invalidates.
        //
        // These will not be accessed unless there is a GPU VA space registered
        // on a GPU under this NPU. They are protected by bit locks in the locks
        // field.
        __be64 __iomem *io_addrs[UVM_MAX_ATSD_REGS];

        // Actual number of registers in the io_addrs array
        size_t count;

        // Bitmask for allocation and locking of the registers. Bit index n
        // corresponds to io_addrs[n]. A set bit means that index is in use
        // (locked).
        DECLARE_BITMAP(locks, UVM_MAX_ATSD_REGS);

        // Max value of any uvm_parent_gpu_t::num_hshub_tlb_invalidate_membars
        // for all retained GPUs under this NPU.
        NvU32 num_membars;
    } atsd_regs;
} uvm_ibm_npu_t;

#if UVM_IBM_NPU_SUPPORTED()
    NV_STATUS uvm_ibm_add_gpu(uvm_parent_gpu_t *parent_gpu);
    void uvm_ibm_remove_gpu(uvm_parent_gpu_t *parent_gpu);
#else
    static NV_STATUS uvm_ibm_add_gpu(uvm_parent_gpu_t *parent_gpu)
    {
        return NV_OK;
    }

    static void uvm_ibm_remove_gpu(uvm_parent_gpu_t *parent_gpu)
    {

    }
#endif // UVM_IBM_NPU_SUPPORTED

#if UVM_ATS_IBM_SUPPORTED()
    // Enables ATS access for the gpu_va_space on the mm_struct associated with
    // the VA space (va_space_mm).
    //
    // If UVM_ATS_IBM_SUPPORTED_IN_KERNEL() is 1, NV_ERR_NOT_SUPPORTED is
    // returned if current->mm does not match va_space_mm.mm or if a GPU VA
    // space within another VA space has already called this function on the
    // same mm.
    //
    // If UVM_ATS_IBM_SUPPORTED_IN_DRIVER() is 1 there are no such restrictions.
    //
    // LOCKING: The VA space lock must be held in write mode.
    //          current->mm->mmap_lock must be held in write mode iff
    //          UVM_ATS_IBM_SUPPORTED_IN_KERNEL() is 1.
    NV_STATUS uvm_ats_ibm_register_gpu_va_space(uvm_gpu_va_space_t *gpu_va_space);

    // Disables ATS access for the gpu_va_space. Prior to calling this function,
    // the caller must guarantee that the GPU will no longer make any ATS
    // accesses in this GPU VA space, and that no ATS fault handling for this
    // GPU will be attempted.
    //
    // LOCKING: This function may block on mmap_lock and the VA space lock, so
    //          neither must be held.
    void uvm_ats_ibm_unregister_gpu_va_space(uvm_gpu_va_space_t *gpu_va_space);

    // Request the kernel to handle a fault.
    //
    // LOCKING: mmap_lock must be held.
    NV_STATUS uvm_ats_ibm_service_fault(uvm_gpu_va_space_t *gpu_va_space,
                                        NvU64 fault_addr,
                                        uvm_fault_access_type_t access_type);

    // Synchronously invalidate ATS translations cached by GPU TLBs. The
    // invalidate applies to all GPUs with active GPU VA spaces in va_space, and
    // covers all pages touching any part of the given range. end is inclusive.
    //
    // GMMU translations in the given range are not guaranteed to be
    // invalidated.
    //
    // LOCKING: No locks are required, but this function may be called with
    //          interrupts disabled.
    void uvm_ats_ibm_invalidate(uvm_va_space_t *va_space, NvU64 start, NvU64 end);
#else
    static NV_STATUS uvm_ats_ibm_register_gpu_va_space(uvm_gpu_va_space_t *gpu_va_space)
    {
        return NV_OK;
    }

    static void uvm_ats_ibm_unregister_gpu_va_space(uvm_gpu_va_space_t *gpu_va_space)
    {

    }

    static NV_STATUS uvm_ats_ibm_service_fault(uvm_gpu_va_space_t *gpu_va_space,
                                               NvU64 fault_addr,
                                               uvm_fault_access_type_t access_type)
    {
        return NV_ERR_NOT_SUPPORTED;
    }

    static void uvm_ats_ibm_invalidate(uvm_va_space_t *va_space, NvU64 start, NvU64 end)
    {

    }
#endif // UVM_ATS_IBM_SUPPORTED

#endif // __UVM_ATS_IBM_H__
