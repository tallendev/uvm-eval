/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "nvidia-drm-conftest.h"

#if defined(NV_DRM_AVAILABLE)

#if defined(NV_DRM_DRM_PRIME_H_PRESENT)
#include <drm/drm_prime.h>
#endif

#if defined(NV_DRM_DRMP_H_PRESENT)
#include <drm/drmP.h>
#endif

#if defined(NV_DRM_DRM_DRV_H_PRESENT)
#include <drm/drm_drv.h>
#endif

#include "nvidia-drm-gem-dma-buf.h"
#include "nvidia-drm-ioctl.h"

#include "linux/dma-buf.h"

static inline
void __nv_drm_gem_dma_buf_free(struct nv_drm_gem_object *nv_gem)
{
    struct nv_drm_gem_dma_buf *nv_dma_buf = to_nv_dma_buf(nv_gem);

    drm_prime_gem_destroy(&nv_gem->base, nv_dma_buf->sgt);

    nv_drm_free(nv_dma_buf);
}

static int __nv_drm_gem_dma_buf_create_mmap_offset(
    struct nv_drm_device *nv_dev,
    struct nv_drm_gem_object *nv_gem,
    uint64_t *offset)
{
    (void)nv_dev;
    return nv_drm_gem_create_mmap_offset(nv_gem, offset);
}

static int __nv_drm_gem_dma_buf_mmap(struct nv_drm_gem_object *nv_gem,
                                     struct vm_area_struct *vma)
{
    struct dma_buf_attachment *attach = nv_gem->base.import_attach;
    struct dma_buf *dma_buf = attach->dmabuf;
    struct file *old_file;
    int ret;

    /* check if buffer supports mmap */
    if (!dma_buf->file->f_op->mmap)
        return -EINVAL;

    /* readjust the vma */
    get_file(dma_buf->file);
    old_file = vma->vm_file;
    vma->vm_file = dma_buf->file;
    vma->vm_pgoff -= drm_vma_node_start(&nv_gem->base.vma_node);;

    ret = dma_buf->file->f_op->mmap(dma_buf->file, vma);

    if (ret) {
        /* restore old parameters on failure */
        vma->vm_file = old_file;
        fput(dma_buf->file);
    } else {
        if (old_file)
            fput(old_file);
    }

    return ret;
}

const struct nv_drm_gem_object_funcs __nv_gem_dma_buf_ops = {
    .free = __nv_drm_gem_dma_buf_free,
    .create_mmap_offset = __nv_drm_gem_dma_buf_create_mmap_offset,
    .mmap = __nv_drm_gem_dma_buf_mmap,
};

struct drm_gem_object*
nv_drm_gem_prime_import_sg_table(struct drm_device *dev,
                                 struct dma_buf_attachment *attach,
                                 struct sg_table *sgt)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    struct dma_buf *dma_buf = attach->dmabuf;
    struct nv_drm_gem_dma_buf *nv_dma_buf;
    struct NvKmsKapiMemory *pMemory;

    if ((nv_dma_buf =
            nv_drm_calloc(1, sizeof(*nv_dma_buf))) == NULL) {
        return NULL;
    }

    // dma_buf->size must be a multiple of PAGE_SIZE
    BUG_ON(dma_buf->size % PAGE_SIZE);

    pMemory = NULL;
#if defined(NV_DRM_ATOMIC_MODESET_AVAILABLE)
    if (drm_core_check_feature(dev, DRIVER_MODESET)) {
        pMemory = nvKms->getSystemMemoryHandleFromDmaBuf(nv_dev->pDevice,
                                                  (NvP64)(NvUPtr)dma_buf,
                                                  dma_buf->size - 1);
    }
#endif

    nv_drm_gem_object_init(nv_dev, &nv_dma_buf->base,
                           &__nv_gem_dma_buf_ops, dma_buf->size, pMemory);

    nv_dma_buf->sgt = sgt;

    return &nv_dma_buf->base.base;
}

#endif
