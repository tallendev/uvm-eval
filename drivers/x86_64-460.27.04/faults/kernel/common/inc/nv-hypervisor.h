/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2018 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#ifndef _NV_HYPERVISOR_H_
#define _NV_HYPERVISOR_H_

#include <nv-kernel-interface-api.h>

// Enums for supported hypervisor types.
// New hypervisor type should be added before OS_HYPERVISOR_CUSTOM_FORCED
typedef enum _HYPERVISOR_TYPE
{
    OS_HYPERVISOR_XEN = 0,
    OS_HYPERVISOR_VMWARE,
    OS_HYPERVISOR_HYPERV,
    OS_HYPERVISOR_KVM,
    OS_HYPERVISOR_PARALLELS,
    OS_HYPERVISOR_CUSTOM_FORCED,
    OS_HYPERVISOR_UNKNOWN
} HYPERVISOR_TYPE;

#define CMD_VGPU_VFIO_WAKE_WAIT_QUEUE         0
#define CMD_VGPU_VFIO_INJECT_INTERRUPT        1
#define CMD_VGPU_VFIO_REGISTER_MDEV           2
#define CMD_VGPU_VFIO_PRESENT                 3

#define MAX_VF_COUNT_PER_GPU 64

typedef enum _VGPU_TYPE_INFO
{
    VGPU_TYPE_NAME = 0,
    VGPU_TYPE_DESCRIPTION,
    VGPU_TYPE_INSTANCES,
} VGPU_TYPE_INFO;

typedef struct
{
    void  *vgpuVfioRef;
    void  *waitQueue;
    void  *nv;
    NvU32 *vgpuTypeIds;
    NvU32  numVgpuTypes;
    NvU32  domain;
    NvU8   bus;
    NvU8   slot;
    NvU8   function;
    NvBool is_virtfn;
} vgpu_vfio_info;

typedef struct
{
    NvU32       domain;
    NvU8        bus;
    NvU8        slot;
    NvU8        function;
    NvBool      isNvidiaAttached;
    NvBool      isMdevAttached;
} vgpu_vf_pci_info;

typedef enum VGPU_CMD_PROCESS_VF_INFO_E
{
    NV_VGPU_SAVE_VF_INFO         = 0,
    NV_VGPU_REMOVE_VF_PCI_INFO   = 1,
    NV_VGPU_REMOVE_VF_MDEV_INFO  = 2,
    NV_VGPU_GET_VF_INFO          = 3
} VGPU_CMD_PROCESS_VF_INFO;

typedef enum VGPU_DEVICE_STATE_E
{
    NV_VGPU_DEV_UNUSED = 0,
    NV_VGPU_DEV_OPENED = 1,
    NV_VGPU_DEV_IN_USE = 2
} VGPU_DEVICE_STATE;

typedef enum _VMBUS_CMD_TYPE
{
    VMBUS_CMD_TYPE_INVALID    = 0,
    VMBUS_CMD_TYPE_SETUP      = 1,
    VMBUS_CMD_TYPE_SENDPACKET = 2,
    VMBUS_CMD_TYPE_CLEANUP    = 3,
} VMBUS_CMD_TYPE;

typedef struct
{
    NvU32 request_id;
    NvU32 page_count;
    NvU64 *pPfns;
    void *buffer;
    NvU32 bufferlen;
} vmbus_send_packet_cmd_params;


typedef struct
{
    NvU32 override_sint;
    NvU8 *nv_guid;
} vmbus_setup_cmd_params;

/*
 * Function prototypes
 */

HYPERVISOR_TYPE NV_API_CALL nv_get_hypervisor_type(void);

#endif // _NV_HYPERVISOR_H_
