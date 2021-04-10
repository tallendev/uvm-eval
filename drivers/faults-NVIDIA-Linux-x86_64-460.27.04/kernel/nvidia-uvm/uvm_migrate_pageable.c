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

#include "uvm_common.h"
#include "uvm_linux.h"
#include "uvm_gpu.h"
#include "uvm_lock.h"
#include "uvm_va_space.h"
#include "uvm_va_range.h"
#include "uvm_tracker.h"
#include "uvm_api.h"
#include "uvm_push.h"
#include "uvm_hal.h"
#include "uvm_migrate_pageable.h"
#include "uvm_populate_pageable.h"

#include <linux/migrate.h>

#if defined(CONFIG_MIGRATE_VMA_HELPER)

static struct kmem_cache *g_uvm_migrate_vma_state_cache __read_mostly;

static const gfp_t g_migrate_vma_gfp_flags = NV_UVM_GFP_FLAGS | GFP_HIGHUSER_MOVABLE | __GFP_THISNODE;

// The calls to migrate_vma are capped at 32MB to set an upper bound on the
// amount of metadata that needs to be allocated for the operation. This number
// was chosen because performance seems to plateau at this size.
#define UVM_MIGRATE_VMA_MAX_SIZE (32UL * 1024 * 1024)
#define UVM_MIGRATE_VMA_MAX_PAGES (UVM_MIGRATE_VMA_MAX_SIZE / PAGE_SIZE)

typedef struct
{
    // Input parameters
    //
    uvm_va_space_t     *va_space;
    uvm_processor_id_t  dst_id;
    int                 dst_node_id;

    // Output parameters
    //
    // Error code. This only signals errors in internal UVM operations.
    // Pages that failed allocation or could not be populated are communicated
    // using the fields below.
    NV_STATUS           status;

    // Whether the masks below must be examined by the caller
    bool                unpopulated_pages : 1;
    bool                allocation_failed  : 1;

    // Mask of pages that couldn't be made resident on the destination because
    // (a) they are backed with data but pages are not populated (e.g. in swap),
    // (b) pages are not backed with any data yet but were not populated
    // due to the vma not being READ_WRITE, as it would not charge the pages to
    // the process properly.
    DECLARE_BITMAP(unpopulated_pages_mask, UVM_MIGRATE_VMA_MAX_PAGES);

    // Mask of pages that failed allocation on the destination
    DECLARE_BITMAP(allocation_failed_mask, UVM_MIGRATE_VMA_MAX_PAGES);

    // Global state managed by the caller
    //
    // These are scratch masks that can be used by the migrate_vma caller to
    // save output page masks and orchestrate the migrate_vma
    // retries/population calls if needed.
    DECLARE_BITMAP(scratch1_mask, UVM_MIGRATE_VMA_MAX_PAGES);
    DECLARE_BITMAP(scratch2_mask, UVM_MIGRATE_VMA_MAX_PAGES);

    // Arrays used by migrate_vma to store the src/dst pfns
    unsigned long dst_pfn_array[UVM_MIGRATE_VMA_MAX_PAGES];
    unsigned long src_pfn_array[UVM_MIGRATE_VMA_MAX_PAGES];

    // Internal state
    //
    uvm_tracker_t tracker;

    DECLARE_BITMAP(pending_page_mask, UVM_MIGRATE_VMA_MAX_PAGES);

    struct {
        // Array of page IOMMU mappings created during allocate_and_copy.
        // Required when using SYS aperture. They are freed in
        // finalize_and_map. Also keep an array with the GPUs for which the
        // mapping was created.
        NvU64              addrs[UVM_MIGRATE_VMA_MAX_PAGES];
        uvm_gpu_t    *addrs_gpus[UVM_MIGRATE_VMA_MAX_PAGES];

        // Mask of pages with entries in the dma address arrays above
        DECLARE_BITMAP(page_mask, UVM_MIGRATE_VMA_MAX_PAGES);

        // Number of pages for which IOMMU mapping were created
        unsigned  long num_pages;
    } dma;

    // Processors where pages are resident before calling migrate_vma
    uvm_processor_mask_t src_processors;

    // Array of per-processor page masks with the pages that are resident
    // before calling migrate_vma.
    struct {
        DECLARE_BITMAP(page_mask, UVM_MIGRATE_VMA_MAX_PAGES);
    } processors[UVM_ID_MAX_PROCESSORS];

    // Number of pages in the migrate_vma call
    unsigned long num_pages;

    // Number of pages that are directly populated on the destination
    unsigned long num_populate_anon_pages;
} migrate_vma_state_t;

// Compute the address needed for copying_gpu to access the given page,
// resident on resident_id.
static NV_STATUS migrate_vma_page_copy_address(struct page *page,
                                               unsigned long page_index,
                                               uvm_processor_id_t resident_id,
                                               uvm_gpu_t *copying_gpu,
                                               migrate_vma_state_t *state,
                                               uvm_gpu_address_t *gpu_addr)
{
    uvm_va_space_t *va_space = state->va_space;
    uvm_gpu_t *owning_gpu = UVM_ID_IS_CPU(resident_id)? NULL: uvm_va_space_get_gpu(va_space, resident_id);
    const bool can_copy_from = uvm_processor_mask_test(&va_space->can_copy_from[uvm_id_value(copying_gpu->id)],
                                                       resident_id);
    const bool direct_peer = owning_gpu &&
                             (owning_gpu != copying_gpu) &&
                             can_copy_from &&
                             !uvm_gpu_peer_caps(owning_gpu, copying_gpu)->is_indirect_peer;

    UVM_ASSERT(page_index < state->num_pages);

    memset(gpu_addr, 0, sizeof(*gpu_addr));

    if (owning_gpu == copying_gpu) {
        // Local vidmem address
        *gpu_addr = uvm_gpu_address_from_phys(uvm_gpu_page_to_phys_address(owning_gpu, page));
    }
    else if (direct_peer) {
        // Direct GPU peer
        uvm_gpu_identity_mapping_t *gpu_peer_mappings = uvm_gpu_get_peer_mapping(copying_gpu, owning_gpu->id);
        uvm_gpu_phys_address_t phys_addr = uvm_gpu_page_to_phys_address(owning_gpu, page);

        *gpu_addr = uvm_gpu_address_virtual(gpu_peer_mappings->base + phys_addr.address);
    }
    else {
        // Sysmem/Indirect Peer
        NV_STATUS status = uvm_gpu_map_cpu_page(copying_gpu, page, &state->dma.addrs[page_index]);

        if (status != NV_OK)
            return status;

        state->dma.addrs_gpus[page_index] = copying_gpu;

        if (state->dma.num_pages++ == 0)
            bitmap_zero(state->dma.page_mask, state->num_pages);

        UVM_ASSERT(!test_bit(page_index, state->dma.page_mask));

        __set_bit(page_index, state->dma.page_mask);

        *gpu_addr = uvm_gpu_address_physical(UVM_APERTURE_SYS, state->dma.addrs[page_index]);
    }

    return NV_OK;
}

// Return the GPU identified with the given NUMA node id
static uvm_gpu_t *get_gpu_from_node_id(uvm_va_space_t *va_space, int node_id)
{
    uvm_gpu_t *gpu;

    for_each_va_space_gpu(gpu, va_space) {
        if (uvm_gpu_numa_info(gpu)->node_id == node_id)
            return gpu;
    }

    return NULL;
}

// Create a new push to zero pages on dst_id
static NV_STATUS migrate_vma_zero_begin_push(uvm_va_space_t *va_space,
                                             uvm_processor_id_t dst_id,
                                             uvm_gpu_t *gpu,
                                             unsigned long start,
                                             unsigned long outer,
                                             uvm_push_t *push)
{
    uvm_channel_type_t channel_type;

    if (UVM_ID_IS_CPU(dst_id)) {
        channel_type = UVM_CHANNEL_TYPE_GPU_TO_CPU;
    }
    else {
        UVM_ASSERT(uvm_id_equal(dst_id, gpu->id));
        channel_type = UVM_CHANNEL_TYPE_GPU_INTERNAL;
    }

    return uvm_push_begin(gpu->channel_manager,
                          channel_type,
                          push,
                          "Zero %s from %s VMA region [0x%lx, 0x%lx]",
                          uvm_va_space_processor_name(va_space, dst_id),
                          uvm_va_space_processor_name(va_space, gpu->id),
                          start,
                          outer);
}

// Create a new push to copy pages between src_id and dst_id
static NV_STATUS migrate_vma_copy_begin_push(uvm_va_space_t *va_space,
                                             uvm_processor_id_t dst_id,
                                             uvm_processor_id_t src_id,
                                             unsigned long start,
                                             unsigned long outer,
                                             uvm_push_t *push)
{
    uvm_channel_type_t channel_type;
    uvm_gpu_t *gpu;

    UVM_ASSERT_MSG(!uvm_id_equal(src_id, dst_id),
                   "Unexpected copy to self, processor %s\n",
                   uvm_va_space_processor_name(va_space, src_id));

    if (UVM_ID_IS_CPU(src_id)) {
        gpu = uvm_va_space_get_gpu(va_space, dst_id);
        channel_type = UVM_CHANNEL_TYPE_CPU_TO_GPU;
    }
    else if (UVM_ID_IS_CPU(dst_id)) {
        gpu = uvm_va_space_get_gpu(va_space, src_id);
        channel_type = UVM_CHANNEL_TYPE_GPU_TO_CPU;
    }
    else {
        // For GPU to GPU copies, prefer to "push" the data from the source as
        // that works better
        gpu = uvm_va_space_get_gpu(va_space, src_id);

        channel_type = UVM_CHANNEL_TYPE_GPU_TO_GPU;
    }

    // NUMA-enabled GPUs can copy to any other NUMA node in the system even if
    // P2P access has not been explicitly enabled (ie va_space->can_copy_from
    // is not set).
    if (!gpu->parent->numa_info.enabled) {
        UVM_ASSERT_MSG(uvm_processor_mask_test(&va_space->can_copy_from[uvm_id_value(gpu->id)], dst_id),
                       "GPU %s dst %s src %s\n",
                       uvm_va_space_processor_name(va_space, gpu->id),
                       uvm_va_space_processor_name(va_space, dst_id),
                       uvm_va_space_processor_name(va_space, src_id));
        UVM_ASSERT_MSG(uvm_processor_mask_test(&va_space->can_copy_from[uvm_id_value(gpu->id)], src_id),
                       "GPU %s dst %s src %s\n",
                       uvm_va_space_processor_name(va_space, gpu->id),
                       uvm_va_space_processor_name(va_space, dst_id),
                       uvm_va_space_processor_name(va_space, src_id));
    }

    if (channel_type == UVM_CHANNEL_TYPE_GPU_TO_GPU) {
        uvm_gpu_t *dst_gpu = uvm_va_space_get_gpu(va_space, dst_id);
        return uvm_push_begin_gpu_to_gpu(gpu->channel_manager,
                                         dst_gpu,
                                         push,
                                         "Copy from %s to %s for VMA region [0x%lx, 0x%lx]",
                                         uvm_va_space_processor_name(va_space, src_id),
                                         uvm_va_space_processor_name(va_space, dst_id),
                                         start,
                                         outer);
    }

    return uvm_push_begin(gpu->channel_manager,
                          channel_type,
                          push,
                          "Copy from %s to %s for VMA region [0x%lx, 0x%lx]",
                          uvm_va_space_processor_name(va_space, src_id),
                          uvm_va_space_processor_name(va_space, dst_id),
                          start,
                          outer);
}

static void migrate_vma_compute_masks(struct vm_area_struct *vma,
                                      const unsigned long *src,
                                      migrate_vma_state_t *state)
{
    unsigned long i;
    const bool is_rw = vma->vm_flags & VM_WRITE;

    UVM_ASSERT(vma_is_anonymous(vma));

    bitmap_fill(state->pending_page_mask, state->num_pages);

    bitmap_zero(state->unpopulated_pages_mask, state->num_pages);
    bitmap_zero(state->allocation_failed_mask, state->num_pages);

    uvm_processor_mask_zero(&state->src_processors);
    state->num_populate_anon_pages = 0;
    state->dma.num_pages = 0;

    for (i = 0; i < state->num_pages; ++i) {
        uvm_processor_id_t src_id;
        struct page *src_page = NULL;
        int src_nid;
        uvm_gpu_t *src_gpu = NULL;

        // Skip pages that cannot be migrated
        if (!(src[i] & MIGRATE_PFN_MIGRATE)) {
            __clear_bit(i, state->pending_page_mask);

            // Page is not populated, likely on swap. Signal the caller to
            // populate first, and retry migrate_vma.
            if (!(src[i] & MIGRATE_PFN_VALID)) {
                __set_bit(i, state->unpopulated_pages_mask);
                state->unpopulated_pages = true;
            }

            continue;
        }

        src_page = migrate_pfn_to_page(src[i]);
        if (!src_page) {
            if (is_rw) {
                // Populate PROT_WRITE vmas in migrate_vma so we can use the
                // GPU's copy engines
                if (state->num_populate_anon_pages++ == 0)
                    bitmap_zero(state->processors[uvm_id_value(state->dst_id)].page_mask, state->num_pages);

                __set_bit(i, state->processors[uvm_id_value(state->dst_id)].page_mask);
            }
            else {
                __clear_bit(i, state->pending_page_mask);

                // PROT_NONE vmas cannot be populated. PROT_READ anonymous vmas
                // are populated using the zero page. In order to match this
                // behavior, we tell the caller to populate using
                // get_user_pages.
                __set_bit(i, state->unpopulated_pages_mask);
                state->unpopulated_pages = true;
            }

            continue;
        }

        src_nid = page_to_nid(src_page);

        // Already at destination
        if (src_nid == state->dst_node_id) {
            __clear_bit(i, state->pending_page_mask);
            continue;
        }

        // Already resident on a CPU node, don't move
        if (UVM_ID_IS_CPU(state->dst_id) && node_state(src_nid, N_CPU)) {
            __clear_bit(i, state->pending_page_mask);
            continue;
        }

        src_gpu = get_gpu_from_node_id(state->va_space, src_nid);

        // Already resident on a node with no CPUs that doesn't belong to a
        // GPU, don't move
        if (UVM_ID_IS_CPU(state->dst_id) && !src_gpu) {
            __clear_bit(i, state->pending_page_mask);
            continue;
        }

        // TODO: Bug 2449272: Implement non-P2P copies. All systems that hit
        // this path have P2P copy support between all GPUs in the system, but
        // it could change in the future.

        if (src_gpu)
            src_id = src_gpu->id;
        else
            src_id = UVM_ID_CPU;

        if (!uvm_processor_mask_test_and_set(&state->src_processors, src_id))
            bitmap_zero(state->processors[uvm_id_value(src_id)].page_mask, state->num_pages);

        __set_bit(i, state->processors[uvm_id_value(src_id)].page_mask);
    }
}

static struct page *migrate_vma_alloc_page(migrate_vma_state_t *state)
{
    struct page *dst_page;
    uvm_va_space_t *va_space = state->va_space;

    if (uvm_enable_builtin_tests && atomic_dec_if_positive(&va_space->test.migrate_vma_allocation_fail_nth) == 0) {
        dst_page = NULL;
    }
    else {
        dst_page = alloc_pages_node(state->dst_node_id, g_migrate_vma_gfp_flags, 0);

        // TODO: Bug 2399573: Linux commit
        // 183f6371aac2a5496a8ef2b0b0a68562652c3cdb introduced a bug that makes
        // __GFP_THISNODE not always be honored (this was later fixed in commit
        // 7810e6781e0fcbca78b91cf65053f895bf59e85f). Therefore, we verify
        // whether the flag has been honored and abort the allocation,
        // otherwise. Remove this check when the fix is deployed on all
        // production systems.
        if (dst_page && page_to_nid(dst_page) != state->dst_node_id) {
            __free_page(dst_page);
            dst_page = NULL;
        }
    }

    return dst_page;
}

static NV_STATUS migrate_vma_populate_anon_pages(struct vm_area_struct *vma,
                                                 unsigned long *dst,
                                                 unsigned long start,
                                                 unsigned long outer,
                                                 migrate_vma_state_t *state)
{
    NV_STATUS status = NV_OK;
    unsigned long i;
    unsigned long *page_mask = state->processors[uvm_id_value(state->dst_id)].page_mask;
    uvm_push_t push;
    uvm_gpu_t *copying_gpu = NULL;
    uvm_va_space_t *va_space = state->va_space;

    // Nothing to do
    if (state->num_populate_anon_pages == 0)
        return NV_OK;

    UVM_ASSERT(state->num_populate_anon_pages == bitmap_weight(page_mask, state->num_pages));

    for_each_set_bit(i, page_mask, state->num_pages) {
        uvm_gpu_address_t dst_address;
        struct page *dst_page;

        __clear_bit(i, state->pending_page_mask);

        dst_page = migrate_vma_alloc_page(state);
        if (!dst_page) {
            __set_bit(i, state->allocation_failed_mask);
            state->allocation_failed = true;
            continue;
        }

        if (!copying_gpu) {
            // Try to get a GPU attached to the node being populated. If there
            // is none, use any of the GPUs registered in the VA space.
            if (UVM_ID_IS_CPU(state->dst_id)) {
                copying_gpu = uvm_va_space_find_first_gpu_attached_to_cpu_node(va_space, state->dst_node_id);
                if (!copying_gpu)
                    copying_gpu = uvm_va_space_find_first_gpu(va_space);
            }
            else {
                copying_gpu = uvm_va_space_get_gpu(va_space, state->dst_id);
            }

            UVM_ASSERT(copying_gpu);

            status = migrate_vma_zero_begin_push(va_space, state->dst_id, copying_gpu, start, outer - 1, &push);
            if (status != NV_OK) {
                __free_page(dst_page);
                return status;
            }
        }
        else {
            uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_PIPELINED);
        }

        status = migrate_vma_page_copy_address(dst_page, i, state->dst_id, copying_gpu, state, &dst_address);
        if (status != NV_OK) {
            __free_page(dst_page);
            break;
        }

        lock_page(dst_page);

        // We'll push one membar later for all memsets in this loop
        uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_NONE);
        copying_gpu->parent->ce_hal->memset_8(&push, dst_address, 0, PAGE_SIZE);

        dst[i] = migrate_pfn(page_to_pfn(dst_page)) | MIGRATE_PFN_LOCKED;
    }

    if (copying_gpu) {
        NV_STATUS tracker_status;

        uvm_push_end(&push);

        tracker_status = uvm_tracker_add_push_safe(&state->tracker, &push);
        if (status == NV_OK)
            status = tracker_status;
    }

    return status;
}

static NV_STATUS migrate_vma_copy_pages_from(struct vm_area_struct *vma,
                                             const unsigned long *src,
                                             unsigned long *dst,
                                             unsigned long start,
                                             unsigned long outer,
                                             uvm_processor_id_t src_id,
                                             migrate_vma_state_t *state)
{
    NV_STATUS status = NV_OK;
    uvm_push_t push;
    uvm_gpu_t *copying_gpu = NULL;
    unsigned long i;
    unsigned long *page_mask = state->processors[uvm_id_value(src_id)].page_mask;
    uvm_va_space_t *va_space = state->va_space;

    UVM_ASSERT(!bitmap_empty(page_mask, state->num_pages));

    for_each_set_bit(i, page_mask, state->num_pages) {
        uvm_gpu_address_t src_address;
        uvm_gpu_address_t dst_address;
        struct page *src_page = migrate_pfn_to_page(src[i]);
        struct page *dst_page;

        UVM_ASSERT(src[i] & MIGRATE_PFN_VALID);
        UVM_ASSERT(src_page);
        UVM_ASSERT(test_bit(i, state->pending_page_mask));

        __clear_bit(i, state->pending_page_mask);

        dst_page = migrate_vma_alloc_page(state);
        if (!dst_page) {
            __set_bit(i, state->allocation_failed_mask);
            state->allocation_failed = true;
            continue;
        }

        if (!copying_gpu) {
            status = migrate_vma_copy_begin_push(va_space, state->dst_id, src_id, start, outer - 1, &push);
            if (status != NV_OK) {
                __free_page(dst_page);
                return status;
            }

            copying_gpu = uvm_push_get_gpu(&push);
        }
        else {
            uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_PIPELINED);
        }

        // We don't have a case where both src and dst use the SYS aperture, so
        // the second call can't overwrite a dma addr set up by the first call.
        status = migrate_vma_page_copy_address(src_page, i, src_id, copying_gpu, state, &src_address);
        if (status == NV_OK)
            status = migrate_vma_page_copy_address(dst_page, i, state->dst_id, copying_gpu, state, &dst_address);

        if (status != NV_OK) {
            __free_page(dst_page);
            break;
        }

        lock_page(dst_page);

        // We'll push one membar later for all copies in this loop
        uvm_push_set_flag(&push, UVM_PUSH_FLAG_CE_NEXT_MEMBAR_NONE);
        copying_gpu->parent->ce_hal->memcopy(&push, dst_address, src_address, PAGE_SIZE);

        dst[i] = migrate_pfn(page_to_pfn(dst_page)) | MIGRATE_PFN_LOCKED;
    }

    // TODO: Bug 1766424: If the destination is a GPU and the copy was done by
    //       that GPU, use a GPU-local membar if no peer nor the CPU can
    //       currently map this page. When peer access gets enabled, do a
    //       MEMBAR_SYS at that point.
    if (copying_gpu) {
        NV_STATUS tracker_status;

        uvm_push_end(&push);

        tracker_status = uvm_tracker_add_push_safe(&state->tracker, &push);
        if (status == NV_OK)
            status = tracker_status;
    }

    return status;
}

static NV_STATUS migrate_vma_copy_pages(struct vm_area_struct *vma,
                                        const unsigned long *src,
                                        unsigned long *dst,
                                        unsigned long start,
                                        unsigned long outer,
                                        migrate_vma_state_t *state)
{
    uvm_processor_id_t src_id;

    for_each_id_in_mask(src_id, &state->src_processors) {
        NV_STATUS status = migrate_vma_copy_pages_from(vma, src, dst, start, outer, src_id, state);
        if (status != NV_OK)
            return status;
    }

    UVM_ASSERT(bitmap_empty(state->pending_page_mask, state->num_pages));

    return NV_OK;
}

static void migrate_vma_alloc_and_copy(struct vm_area_struct *vma,
                                       const unsigned long *src,
                                       unsigned long *dst,
                                       unsigned long start,
                                       unsigned long outer,
                                       void *private)
{
    NV_STATUS tracker_status;
    migrate_vma_state_t *state = (migrate_vma_state_t *)private;

    uvm_tracker_init(&state->tracker);

    state->num_pages = (outer - start) / PAGE_SIZE;
    state->status = NV_OK;
    state->unpopulated_pages = false;
    state->allocation_failed = false;

    migrate_vma_compute_masks(vma, src, state);

    state->status = migrate_vma_populate_anon_pages(vma, dst, start, outer, state);

    if (state->status == NV_OK)
        state->status = migrate_vma_copy_pages(vma, src, dst, start, outer, state);

    // Wait for tracker since all copies must have completed before returning
    tracker_status = uvm_tracker_wait_deinit(&state->tracker);

    if (state->status == NV_OK)
        state->status = tracker_status;
}

static void migrate_vma_finalize_and_map(struct vm_area_struct *vma,
                                         const unsigned long *src,
                                         const unsigned long *dst,
                                         unsigned long start,
                                         unsigned long outer,
                                         void *private)
{
    migrate_vma_state_t *state = (migrate_vma_state_t *)private;

    // Pages could fail migration if the page state could not be migrated by the
    // kernel. However, when this happens this likely means that the page is
    // populated somewhere, so we do nothing about it.

    // Remove the IOMMU mappings created during the copy
    if (state->dma.num_pages > 0) {
        unsigned long i;

        for_each_set_bit(i, state->dma.page_mask, state->num_pages)
            uvm_gpu_unmap_cpu_page(state->dma.addrs_gpus[i], state->dma.addrs[i]);
    }

    if (state->unpopulated_pages)
        UVM_ASSERT(!bitmap_empty(state->unpopulated_pages_mask, state->num_pages));

    if (state->allocation_failed)
        UVM_ASSERT(!bitmap_empty(state->allocation_failed_mask, state->num_pages));

    UVM_ASSERT(!bitmap_intersects(state->unpopulated_pages_mask,
                                  state->allocation_failed_mask,
                                  state->num_pages));
}

static struct migrate_vma_ops g_migrate_vma_ops =
{
    .alloc_and_copy = migrate_vma_alloc_and_copy,
    .finalize_and_map = migrate_vma_finalize_and_map,
};

static NV_STATUS migrate_pageable_vma_populate_mask(struct vm_area_struct *vma,
                                                    unsigned long start,
                                                    unsigned long outer,
                                                    const unsigned long *mask)
{
    const unsigned long num_pages = (outer - start) / PAGE_SIZE;
    unsigned long subregion_first = find_first_bit(mask, num_pages);

    while (subregion_first < num_pages) {
        NV_STATUS status;
        unsigned long subregion_outer = find_next_zero_bit(mask, num_pages, subregion_first + 1);

        status = uvm_populate_pageable_vma(vma,
                                           start + subregion_first * PAGE_SIZE,
                                           (subregion_outer - subregion_first) * PAGE_SIZE,
                                           0);
        if (status != NV_OK)
            return status;

        subregion_first = find_next_bit(mask, num_pages, subregion_outer + 1);
    }

    return NV_OK;
}

static NV_STATUS migrate_pageable_vma_migrate_mask(struct vm_area_struct *vma,
                                                   unsigned long start,
                                                   unsigned long outer,
                                                   const unsigned long *mask,
                                                   migrate_vma_state_t *migrate_vma_state)
{
    const unsigned long num_pages = (outer - start) / PAGE_SIZE;
    unsigned long subregion_first = find_first_bit(mask, num_pages);

    while (subregion_first < num_pages) {
        unsigned long subregion_outer = find_next_zero_bit(mask, num_pages, subregion_first + 1);
        int ret = migrate_vma(&g_migrate_vma_ops,
                              vma,
                              start + subregion_first * PAGE_SIZE,
                              start + subregion_outer * PAGE_SIZE,
                              migrate_vma_state->src_pfn_array,
                              migrate_vma_state->dst_pfn_array,
                              migrate_vma_state);
        if (ret < 0)
            return errno_to_nv_status(ret);

        if (migrate_vma_state->status != NV_OK)
            return migrate_vma_state->status;

        // We ignore allocation failure here as we are just retrying migration,
        // but pages must have already been populated by the caller

        subregion_first = find_next_bit(mask, num_pages, subregion_outer + 1);
    }

    return NV_OK;
}


static NV_STATUS migrate_pageable_vma_region(struct vm_area_struct *vma,
                                             unsigned long start,
                                             unsigned long outer,
                                             migrate_vma_state_t *migrate_vma_state,
                                             unsigned long *next_addr)
{
    NV_STATUS status;
    const unsigned long num_pages = (outer - start) / PAGE_SIZE;
    struct mm_struct *mm = vma->vm_mm;
    int ret;
    bool unpopulated_pages;
    bool allocation_failed;

    UVM_ASSERT(PAGE_ALIGNED(start));
    UVM_ASSERT(PAGE_ALIGNED(outer));
    UVM_ASSERT(start < outer);
    UVM_ASSERT(start >= vma->vm_start);
    UVM_ASSERT(outer <= vma->vm_end);
    UVM_ASSERT(outer - start <= UVM_MIGRATE_VMA_MAX_SIZE);
    uvm_assert_mmap_lock_locked(mm);
    uvm_assert_rwsem_locked(&migrate_vma_state->va_space->lock);

    ret = migrate_vma(&g_migrate_vma_ops,
                      vma,
                      start,
                      outer,
                      migrate_vma_state->src_pfn_array,
                      migrate_vma_state->dst_pfn_array,
                      migrate_vma_state);
    if (ret < 0)
        return errno_to_nv_status(ret);

    if (migrate_vma_state->status != NV_OK)
        return migrate_vma_state->status;

    unpopulated_pages = migrate_vma_state->unpopulated_pages;
    allocation_failed = migrate_vma_state->allocation_failed;

    // Save the returned page masks if needed
    if (unpopulated_pages)
        bitmap_copy(migrate_vma_state->scratch1_mask, migrate_vma_state->unpopulated_pages_mask, num_pages);

    if (allocation_failed)
        bitmap_copy(migrate_vma_state->scratch2_mask, migrate_vma_state->allocation_failed_mask, num_pages);

    if (unpopulated_pages) {
        // Populate pages using get_user_pages
        status = migrate_pageable_vma_populate_mask(vma, start, outer, migrate_vma_state->scratch1_mask);
        if (status != NV_OK)
            return status;

        // Retry migration
        status = migrate_pageable_vma_migrate_mask(vma,
                                                   start,
                                                   outer,
                                                   migrate_vma_state->scratch1_mask,
                                                   migrate_vma_state);
        if (status != NV_OK)
            return status;
    }

    if (allocation_failed) {
        // If the destination is the CPU, signal user-space to retry with a
        // different node. Otherwise, just try to populate anywhere in the
        // system
        if (UVM_ID_IS_CPU(migrate_vma_state->dst_id)) {
            *next_addr = start + find_first_bit(migrate_vma_state->scratch2_mask, num_pages) * PAGE_SIZE;
            return NV_ERR_MORE_PROCESSING_REQUIRED;
        }
        else {
            status = migrate_pageable_vma_populate_mask(vma, start, outer, migrate_vma_state->scratch2_mask);
            if (status != NV_OK)
                return status;
        }
    }

    return NV_OK;
}

static NV_STATUS migrate_pageable_vma(struct vm_area_struct *vma,
                                      unsigned long start,
                                      unsigned long outer,
                                      migrate_vma_state_t *migrate_vma_state,
                                      unsigned long *next_addr)
{
    NV_STATUS status = NV_OK;
    struct mm_struct *mm = vma->vm_mm;
    uvm_va_space_t *va_space = migrate_vma_state->va_space;

    UVM_ASSERT(PAGE_ALIGNED(start));
    UVM_ASSERT(PAGE_ALIGNED(outer));
    UVM_ASSERT(vma->vm_end > start);
    UVM_ASSERT(vma->vm_start < outer);
    uvm_assert_mmap_lock_locked(mm);
    uvm_assert_rwsem_locked(&va_space->lock);

    // Adjust to input range boundaries
    start = max(start, vma->vm_start);
    outer = min(outer, vma->vm_end);

    // TODO: Bug 2419180: support file-backed pages in migrate_vma, when
    //       support for it is added to the Linux kernel
    if (!vma_is_anonymous(vma))
        return NV_WARN_NOTHING_TO_DO;

    // If no GPUs are registered, fall back to user space.
    if (uvm_processor_mask_empty(&va_space->registered_gpus))
        return NV_WARN_NOTHING_TO_DO;

    while (start < outer) {
        const size_t region_size = min(outer - start, UVM_MIGRATE_VMA_MAX_SIZE);

        status = migrate_pageable_vma_region(vma,
                                             start,
                                             start + region_size,
                                             migrate_vma_state,
                                             next_addr);
        if (status == NV_ERR_MORE_PROCESSING_REQUIRED) {
            UVM_ASSERT(*next_addr >= start);
            UVM_ASSERT(*next_addr < outer);
        }

        if (status != NV_OK)
            break;

        start += region_size;
    };

    return status;
}

static NV_STATUS migrate_pageable(struct mm_struct *mm,
                                  const unsigned long start,
                                  const unsigned long length,
                                  migrate_vma_state_t * migrate_vma_state,
                                  NvU64 *user_space_start,
                                  NvU64 *user_space_length)
{
    struct vm_area_struct *vma;
    const unsigned long outer = start + length;
    unsigned long prev_outer = outer;
    uvm_va_space_t *va_space = migrate_vma_state->va_space;

    UVM_ASSERT(PAGE_ALIGNED(start));
    UVM_ASSERT(PAGE_ALIGNED(length));
    uvm_assert_mmap_lock_locked(mm);

    vma = find_vma_intersection(mm, start, outer);

    // VMAs are validated and migrated one at a time, since migrate_vma works
    // on one vma at a time
    for (; vma && (vma->vm_start <= prev_outer); vma = vma->vm_next) {
        unsigned long next_addr = 0;
        NV_STATUS status = migrate_pageable_vma(vma,
                                                start,
                                                outer,
                                                migrate_vma_state,
                                                &next_addr);
        if (status == NV_WARN_NOTHING_TO_DO) {
            NV_STATUS populate_status = NV_OK;

            UVM_ASSERT(!vma_is_anonymous(vma) || uvm_processor_mask_empty(&va_space->registered_gpus));

            populate_status = uvm_populate_pageable_vma(vma, start, outer - start, 0);
            if (populate_status == NV_OK) {
                *user_space_start = max(vma->vm_start, start);
                *user_space_length = min(vma->vm_end, outer) - *user_space_start;
            }
            else {
                status = populate_status;
            }
        }
        else if (status == NV_ERR_MORE_PROCESSING_REQUIRED) {
            UVM_ASSERT(next_addr >= start);
            UVM_ASSERT(next_addr < outer);
            UVM_ASSERT(UVM_ID_IS_CPU(migrate_vma_state->dst_id));

            *user_space_start = next_addr;
        }

        if (status != NV_OK)
            return status;

        if (vma->vm_end >= outer)
            return NV_OK;

        prev_outer = vma->vm_end;
    }

    // Input range not fully covered by VMAs.
    return NV_ERR_INVALID_ADDRESS;
}

NV_STATUS uvm_migrate_pageable(uvm_va_space_t *va_space,
                               struct mm_struct *mm,
                               const unsigned long start,
                               const unsigned long length,
                               uvm_processor_id_t dst_id,
                               int dst_cpu_node_id,
                               NvU64 *user_space_start,
                               NvU64 *user_space_length)
{
    migrate_vma_state_t *migrate_vma_state = NULL;
    NV_STATUS status;

    UVM_ASSERT(PAGE_ALIGNED(start));
    UVM_ASSERT(PAGE_ALIGNED(length));
    uvm_assert_mmap_lock_locked(mm);

    // We only check that dst_cpu_node_id is a valid node in the system and it
    // doesn't correspond to a GPU node. This is fine because alloc_pages_node
    // will clamp the allocation to cpuset_current_mems_allowed, and
    // uvm_migrate_pageable is only called from the process context
    // (uvm_migrate). However, this would need to change if we wanted to call
    // this function from a bottom half.
    if (UVM_ID_IS_CPU(dst_id) &&
        (!nv_numa_node_has_memory(dst_cpu_node_id) || get_gpu_from_node_id(va_space, dst_cpu_node_id) != NULL))
        return NV_ERR_INVALID_ARGUMENT;

    migrate_vma_state = kmem_cache_alloc(g_uvm_migrate_vma_state_cache, NV_UVM_GFP_FLAGS);
    if (!migrate_vma_state)
        return NV_ERR_NO_MEMORY;

    migrate_vma_state->va_space    = va_space;
    migrate_vma_state->dst_id      = dst_id;
    migrate_vma_state->dst_node_id = UVM_ID_IS_CPU(dst_id)?
                                         dst_cpu_node_id :
                                         uvm_gpu_numa_info(uvm_va_space_get_gpu(va_space, dst_id))->node_id;

    status = migrate_pageable(mm, start, length, migrate_vma_state, user_space_start, user_space_length);

    kmem_cache_free(g_uvm_migrate_vma_state_cache, migrate_vma_state);

    return status;
}

NV_STATUS uvm_migrate_pageable_init()
{
    g_uvm_migrate_vma_state_cache = NV_KMEM_CACHE_CREATE("migrate_vma_state_t", migrate_vma_state_t);
    if (!g_uvm_migrate_vma_state_cache)
        return NV_ERR_NO_MEMORY;

    return NV_OK;
}

void uvm_migrate_pageable_exit()
{
    kmem_cache_destroy_safe(&g_uvm_migrate_vma_state_cache);
}
#endif
