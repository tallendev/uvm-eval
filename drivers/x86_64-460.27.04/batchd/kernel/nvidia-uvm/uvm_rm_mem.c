/*******************************************************************************
    Copyright (c) 2015-2019 NVIDIA Corporation

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

#include "uvm_rm_mem.h"
#include "uvm_gpu.h"
#include "uvm_global.h"
#include "uvm_kvmalloc.h"
#include "uvm_linux.h"
#include "nv_uvm_interface.h"

NV_STATUS uvm_rm_mem_alloc(uvm_gpu_t *gpu, uvm_rm_mem_type_t type, NvLength size, uvm_rm_mem_t **rm_mem_out)
{
    NV_STATUS status = NV_OK;
    uvm_rm_mem_t *rm_mem = NULL;
    UvmGpuAllocInfo alloc_info = {0};
    NvU64 gpu_va;

    UVM_ASSERT(gpu);

    if (size == 0) {
        UVM_ASSERT(size != 0);
        return NV_ERR_INVALID_ARGUMENT;
    }

    rm_mem = uvm_kvmalloc_zero(sizeof(*rm_mem));
    if (rm_mem == NULL)
        return NV_ERR_NO_MEMORY;

    switch (type) {
        case UVM_RM_MEM_TYPE_SYS:
            status = uvm_rm_locked_call(nvUvmInterfaceMemoryAllocSys(gpu->rm_address_space, size, &gpu_va, &alloc_info));
            break;
        case UVM_RM_MEM_TYPE_GPU:
            status = uvm_rm_locked_call(nvUvmInterfaceMemoryAllocFB(gpu->rm_address_space, size, &gpu_va, &alloc_info));
            break;
        default:
            UVM_ASSERT_MSG(0, "Invalid memory type 0x%x\n", type);
            status = NV_ERR_INVALID_ARGUMENT;
            goto error;
    }

    if (status != NV_OK) {
        UVM_ERR_PRINT("nvUvmInterfaceMemoryAlloc%s() failed: %s, GPU %s\n",
                type == UVM_RM_MEM_TYPE_SYS ? "Sys" : "FB", nvstatusToString(status), uvm_gpu_name(gpu));
        goto error;
    }

    rm_mem->gpu_owner = gpu;
    rm_mem->vas[uvm_global_id_value(gpu->global_id)] = gpu_va;
    uvm_global_processor_mask_set(&rm_mem->mapped_on, gpu->global_id);

    rm_mem->type = type;
    rm_mem->size = size;

    *rm_mem_out = rm_mem;

    return NV_OK;

error:
    uvm_kvfree(rm_mem);
    return status;
}

NV_STATUS uvm_rm_mem_map_cpu(uvm_rm_mem_t *rm_mem)
{
    NV_STATUS status;
    uvm_gpu_t *gpu = rm_mem->gpu_owner;
    NvU64 gpu_va = uvm_rm_mem_get_gpu_va(rm_mem, gpu);
    void *cpu_va;

    UVM_ASSERT(rm_mem);

    if (uvm_global_processor_mask_test(&rm_mem->mapped_on, UVM_GLOBAL_ID_CPU)) {
        // Already mapped
        return NV_OK;
    }

    status = uvm_rm_locked_call(nvUvmInterfaceMemoryCpuMap(gpu->rm_address_space, gpu_va, rm_mem->size, &cpu_va, UVM_PAGE_SIZE_DEFAULT));
    if (status != NV_OK) {
        UVM_ERR_PRINT("nvUvmInterfaceMemoryCpuMap() failed: %s, GPU %s\n", nvstatusToString(status), uvm_gpu_name(gpu));
        return status;
    }
    rm_mem->vas[UVM_GLOBAL_ID_CPU_VALUE] = (NvU64)cpu_va;
    uvm_global_processor_mask_set(&rm_mem->mapped_on, UVM_GLOBAL_ID_CPU);

    return NV_OK;
}

void uvm_rm_mem_unmap_cpu(uvm_rm_mem_t *rm_mem)
{
    UVM_ASSERT(rm_mem);

    if (!uvm_global_processor_mask_test(&rm_mem->mapped_on, UVM_GLOBAL_ID_CPU)) {
        // Already unmapped
        return;
    }

    uvm_rm_locked_call_void(nvUvmInterfaceMemoryCpuUnMap(rm_mem->gpu_owner->rm_address_space, uvm_rm_mem_get_cpu_va(rm_mem)));

    rm_mem->vas[UVM_ID_CPU_VALUE] = 0;
    uvm_global_processor_mask_clear(&rm_mem->mapped_on, UVM_GLOBAL_ID_CPU);
}

NV_STATUS uvm_rm_mem_map_gpu(uvm_rm_mem_t *rm_mem, uvm_gpu_t *gpu)
{
    NV_STATUS status;
    uvm_gpu_t *gpu_owner = rm_mem->gpu_owner;
    NvU64 gpu_owner_va = uvm_rm_mem_get_gpu_va(rm_mem, gpu_owner);
    NvU64 gpu_va;

    // Peer mappings not supported yet
    UVM_ASSERT(rm_mem->type == UVM_RM_MEM_TYPE_SYS);

    if (uvm_global_processor_mask_test(&rm_mem->mapped_on, gpu->global_id)) {
        // Already mapped
        return NV_OK;
    }

    status = uvm_rm_locked_call(nvUvmInterfaceDupAllocation(0, gpu_owner->rm_address_space, gpu_owner_va,
            gpu->rm_address_space, &gpu_va, false));
    if (status != NV_OK) {
        UVM_ERR_PRINT("nvUvmInterfaceDupAllocation() failed: %s, src GPU %s, dest GPU %s\n",
                      nvstatusToString(status),
                      uvm_gpu_name(gpu_owner),
                      uvm_gpu_name(gpu));
        return status;
    }

    rm_mem->vas[uvm_global_id_value(gpu->global_id)] = gpu_va;
    uvm_global_processor_mask_set(&rm_mem->mapped_on, gpu->global_id);

    return NV_OK;
}

void uvm_rm_mem_unmap_gpu(uvm_rm_mem_t *rm_mem, uvm_gpu_t *gpu)
{
    NvU64 *va;

    UVM_ASSERT(rm_mem);
    UVM_ASSERT(gpu);

    // Cannot unmap from the gpu that owns the allocation.
    UVM_ASSERT_MSG(rm_mem->gpu_owner != gpu, "GPU %s\n", uvm_gpu_name(gpu));

    if (!uvm_global_processor_mask_test(&rm_mem->mapped_on, gpu->global_id)) {
        // Already unmapped
        return;
    }

    va = &rm_mem->vas[uvm_global_id_value(gpu->global_id)];

    uvm_rm_locked_call_void(nvUvmInterfaceMemoryFree(gpu->rm_address_space, *va));
    *va = 0;
    uvm_global_processor_mask_clear(&rm_mem->mapped_on, gpu->global_id);
}

void uvm_rm_mem_free(uvm_rm_mem_t *rm_mem)
{
    uvm_global_gpu_id_t gpu_id;
    uvm_gpu_t *gpu_owner;
    NvU64 *va;

    if (rm_mem == NULL)
        return;

    if (uvm_global_processor_mask_test(&rm_mem->mapped_on, UVM_GLOBAL_ID_CPU))
        uvm_rm_mem_unmap_cpu(rm_mem);

    gpu_owner = rm_mem->gpu_owner;

    // Don't use for_each_global_gpu_in_mask() as the owning GPU might be being
    // destroyed and already removed from the global GPU array causing the iteration
    // to stop prematurely.
    for_each_global_gpu_id_in_mask(gpu_id, &rm_mem->mapped_on) {
        if (!uvm_global_id_equal(gpu_id, gpu_owner->global_id))
            uvm_rm_mem_unmap_gpu(rm_mem, uvm_gpu_get(gpu_id));
    }

    va = &rm_mem->vas[uvm_global_id_value(gpu_owner->global_id)];

    uvm_rm_locked_call_void(nvUvmInterfaceMemoryFree(gpu_owner->rm_address_space, *va));
    uvm_global_processor_mask_clear(&rm_mem->mapped_on, gpu_owner->global_id);
    *va = 0;

    UVM_ASSERT_MSG(uvm_global_processor_mask_get_count(&rm_mem->mapped_on) == 0,
                   "Left-over %u mappings in rm_mem\n",
                   uvm_global_processor_mask_get_count(&rm_mem->mapped_on));

    uvm_kvfree(rm_mem);
}

NV_STATUS uvm_rm_mem_alloc_and_map_cpu(uvm_gpu_t *gpu, uvm_rm_mem_type_t type, NvLength size, uvm_rm_mem_t **rm_mem_out)
{
    uvm_rm_mem_t *rm_mem;
    NV_STATUS status = uvm_rm_mem_alloc(gpu, type, size, &rm_mem);
    if (status != NV_OK)
        return status;
    status = uvm_rm_mem_map_cpu(rm_mem);
    if (status != NV_OK)
        goto error;

    *rm_mem_out = rm_mem;

    return NV_OK;

error:
    uvm_rm_mem_free(rm_mem);
    return status;
}

NV_STATUS uvm_rm_mem_map_all_gpus(uvm_rm_mem_t *rm_mem)
{
    uvm_gpu_t *gpu;

    UVM_ASSERT(rm_mem);

    for_each_global_gpu(gpu) {
        NV_STATUS status = uvm_rm_mem_map_gpu(rm_mem, gpu);
        if (status != NV_OK)
            return status;
    }
    return NV_OK;
}

NV_STATUS uvm_rm_mem_alloc_and_map_all(uvm_gpu_t *gpu, uvm_rm_mem_type_t type, NvLength size, uvm_rm_mem_t **rm_mem_out)
{
    uvm_rm_mem_t *rm_mem;

    NV_STATUS status = uvm_rm_mem_alloc_and_map_cpu(gpu, type, size, &rm_mem);
    if (status != NV_OK)
        return status;

    status = uvm_rm_mem_map_all_gpus(rm_mem);
    if (status != NV_OK)
        goto error;

    *rm_mem_out = rm_mem;

    return NV_OK;

error:
    uvm_rm_mem_free(rm_mem);
    return status;
}

static NvU64 uvm_rm_mem_get_processor_va(uvm_rm_mem_t *rm_mem, uvm_global_processor_id_t id)
{
    UVM_ASSERT_MSG(uvm_global_processor_mask_test(&rm_mem->mapped_on, id), "processor id %u\n", uvm_global_id_value(id));

    return rm_mem->vas[uvm_global_id_value(id)];
}

NvU64 uvm_rm_mem_get_gpu_va(uvm_rm_mem_t *rm_mem, uvm_gpu_t *gpu)
{
    return uvm_rm_mem_get_processor_va(rm_mem, gpu->global_id);
}

void *uvm_rm_mem_get_cpu_va(uvm_rm_mem_t *rm_mem)
{
    return (void *)(unsigned long)uvm_rm_mem_get_processor_va(rm_mem, UVM_GLOBAL_ID_CPU);
}
