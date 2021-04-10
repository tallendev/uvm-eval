/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1993-2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#ifndef _NV_MISC_H_
#define _NV_MISC_H_

#include "nvtypes.h"
#include "nvstatus.h"

#if defined(NV_KERNEL_INTERFACE_LAYER) && defined(__FreeBSD__)
  #include <sys/stddef.h> // NULL
#else
  #include <stddef.h>     // NULL
#endif

#endif /* _NV_MISC_H_ */
