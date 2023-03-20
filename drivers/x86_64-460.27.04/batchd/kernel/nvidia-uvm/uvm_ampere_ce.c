/*******************************************************************************
    Copyright (c) 2018-2020 NVIDIA Corporation

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

#include "uvm_hal.h"
#include "uvm_hal_types.h"
#include "clc6b5.h"

#include "clc7b5.h"            


static NvU32 ce_aperture(uvm_aperture_t aperture)
{
    BUILD_BUG_ON(HWCONST(C6B5, SET_SRC_PHYS_MODE, TARGET, LOCAL_FB) !=
                 HWCONST(C6B5, SET_DST_PHYS_MODE, TARGET, LOCAL_FB));
    BUILD_BUG_ON(HWCONST(C6B5, SET_SRC_PHYS_MODE, TARGET, COHERENT_SYSMEM) !=
                 HWCONST(C6B5, SET_DST_PHYS_MODE, TARGET, COHERENT_SYSMEM));
    BUILD_BUG_ON(HWCONST(C6B5, SET_SRC_PHYS_MODE, TARGET, PEERMEM) !=
                 HWCONST(C6B5, SET_DST_PHYS_MODE, TARGET, PEERMEM));

    if (aperture == UVM_APERTURE_SYS) {
        return HWCONST(C6B5, SET_SRC_PHYS_MODE, TARGET, COHERENT_SYSMEM);
    }
    else if (aperture == UVM_APERTURE_VID) {
        return HWCONST(C6B5, SET_SRC_PHYS_MODE, TARGET, LOCAL_FB);
    }
    else {
        return HWCONST(C6B5, SET_SRC_PHYS_MODE, TARGET, PEERMEM) |
               HWVALUE(C6B5, SET_SRC_PHYS_MODE, FLA, 0) |
               HWVALUE(C6B5, SET_SRC_PHYS_MODE, PEER_ID, UVM_APERTURE_PEER_ID(aperture));
    }
}

// Push SET_{SRC,DST}_PHYS mode if needed and return LAUNCH_DMA_{SRC,DST}_TYPE
// flags
NvU32 uvm_hal_ampere_ce_phys_mode(uvm_push_t *push, uvm_gpu_address_t dst, uvm_gpu_address_t src)
{
    NvU32 launch_dma_src_dst_type = 0;

    if (src.is_virtual)
        launch_dma_src_dst_type |= HWCONST(C6B5, LAUNCH_DMA, SRC_TYPE, VIRTUAL);
    else
        launch_dma_src_dst_type |= HWCONST(C6B5, LAUNCH_DMA, SRC_TYPE, PHYSICAL);

    if (dst.is_virtual)
        launch_dma_src_dst_type |= HWCONST(C6B5, LAUNCH_DMA, DST_TYPE, VIRTUAL);
    else
        launch_dma_src_dst_type |= HWCONST(C6B5, LAUNCH_DMA, DST_TYPE, PHYSICAL);

    if (!src.is_virtual && !dst.is_virtual) {
        NV_PUSH_2U(C6B5, SET_SRC_PHYS_MODE, ce_aperture(src.aperture),
                         SET_DST_PHYS_MODE, ce_aperture(dst.aperture));
    }
    else if (!src.is_virtual) {
        NV_PUSH_1U(C6B5, SET_SRC_PHYS_MODE, ce_aperture(src.aperture));
    }
    else if (!dst.is_virtual) {
        NV_PUSH_1U(C6B5, SET_DST_PHYS_MODE, ce_aperture(dst.aperture));
    }

    return launch_dma_src_dst_type;
}


NvU32 uvm_hal_ampere_ce_plc_mode_c7b5(void)
{
    return HWCONST(C7B5, LAUNCH_DMA, DISABLE_PLC, TRUE); 
}

