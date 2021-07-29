/*******************************************************************************
    Copyright (c) 2018 NVIDIA Corporation

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

#ifndef __UVM_MIGRATE_PAGEABLE_H__
#define __UVM_MIGRATE_PAGEABLE_H__

#include "uvm_common.h"
#include "uvm_linux.h"
#include "uvm_populate_pageable.h"

#if defined(CONFIG_MIGRATE_VMA_HELPER)
// Populates the given VA range and tries to migrate all the pages to dst_id.
// If the destination processor is the CPU, the NUMA node in dst_cpu_node_id
// is used. The input VA range must be fully backed by VMAs. This function
// relies on migrate_vma, which was added in Linux 4.14. For kernels that do
// not provide migrate_vma, this function populates the memory using
// get_user_pages and returns NV_WARN_NOTHING_TO_DO to complete the migration
// in user space. user_space_start and user_space_length will contain the full
// input range. If the destination is the CPU and dst_cpu_node_id is full,
// NV_ERR_MORE_PROCESSING_REQUIRED is returned and user-space will call
// UVM_MIGRATE with the next preferred CPU node (if more are available),
// starting at the address specified by user_space_start. If the destination is
// a GPU and a page could not be populated, return NV_ERR_NO_MEMORY. Otherwise,
// return NV_OK. This is fine because UvmMigrate/UvmMigrateAsync only guarantee
// that the memory is populated somewhere in the system, not that pages moved
// to the requested processor.
//
// migrate_vma does not support file-backed vmas yet. If a file-backed vma is
// found, return NV_WARN_NOTHING_TO_DO to fall back to user-mode. In this case
// user_space_start and user_space_length will contain the intersection of the
// vma address range and [start:start + length].
//
// Also, if no GPUs have been registered in the VA space, return
// NV_WARN_NOTHING_TO_DO to fall back to user space to complete the whole
// migration, too.
//
// Locking: mmap_lock must be held in read or write mode
NV_STATUS uvm_migrate_pageable(uvm_va_space_t *va_space,
                               struct mm_struct *mm,
                               const unsigned long start,
                               const unsigned long length,
                               uvm_processor_id_t dst_id,
                               int dst_cpu_node_id,
                               NvU64 *user_space_start,
                               NvU64 *user_space_length);

NV_STATUS uvm_migrate_pageable_init(void);

void uvm_migrate_pageable_exit(void);
#else

static NV_STATUS uvm_migrate_pageable(uvm_va_space_t *va_space,
                                      struct mm_struct *mm,
                                      const unsigned long start,
                                      const unsigned long length,
                                      uvm_processor_id_t dst_id,
                                      int dst_cpu_node_id,
                                      NvU64 *user_space_start,
                                      NvU64 *user_space_length)
{
    NV_STATUS status = uvm_populate_pageable(mm, start, length, 0);

    if (status != NV_OK)
        return status;

    *user_space_start = start;
    *user_space_length = length;

    return NV_WARN_NOTHING_TO_DO;

}

static NV_STATUS uvm_migrate_pageable_init(void)
{
    return NV_OK;
}

static void uvm_migrate_pageable_exit(void)
{
}

#endif

#endif
