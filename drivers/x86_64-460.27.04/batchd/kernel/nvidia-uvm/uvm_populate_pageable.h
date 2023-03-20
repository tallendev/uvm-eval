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

#ifndef __UVM_POPULATE_PAGEABLE_H__
#define __UVM_POPULATE_PAGEABLE_H__

// Populate the pages of the given vma that overlap with the
// [start:start+length) range. If any of the pages was not populated, we return
// NV_ERR_NO_MEMORY.
//
// Locking: vma->vm_mm->mmap_lock must be held in read or write mode
NV_STATUS uvm_populate_pageable_vma(struct vm_area_struct *vma,
                                    unsigned long start,
                                    unsigned long length,
                                    int min_prot);

// Populate all the pages in the given range by calling get_user_pages. The
// range must be fully backed by vmas. If any of the pages was not populated,
// we return NV_ERR_NO_MEMORY.
//
// Locking: mm->mmap_lock must be held in read or write mode
NV_STATUS uvm_populate_pageable(struct mm_struct *mm,
                                unsigned long start,
                                unsigned long length,
                                int min_prot);

#endif
