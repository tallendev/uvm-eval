/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2020 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */


#ifndef _NV_H_
#define _NV_H_



#if !defined(NVLIMITS_H_MISSING)
#include <nvlimits.h>
#else
#define NV_MAX_DEVICES 32
#endif

#include <nvtypes.h>
#include <nvCpuUuid.h>
#include <stdarg.h>
#include <nv-caps.h>
#include <nv-ioctl.h>
#include <nvmisc.h>

extern nv_cap_t *nvidia_caps_root;






#if !defined(NV_KERNEL_INTERFACE_API_H_MISSING)
#include <nv-kernel-interface-api.h>
#endif

/* NVIDIA's reserved major character device number (Linux). */
#define NV_MAJOR_DEVICE_NUMBER 195

#define GPU_UUID_LEN    (16)

/*
 * Buffer size for an ASCII UUID: We need 2 digits per byte, plus space
 * for "GPU", 5 dashes, and '\0' termination:
 */
#define GPU_UUID_ASCII_LEN  (GPU_UUID_LEN * 2 + 9)

/*
 * #define an absolute maximum used as a sanity check for the
 * NV_ESC_IOCTL_XFER_CMD ioctl() size argument.
 */
#define NV_ABSOLUTE_MAX_IOCTL_SIZE  16384

/*
 * Solaris provides no more than 8 bits for the argument size in
 * the ioctl() command encoding; make sure we don't exceed this
 * limit.
 */
#define __NV_IOWR_ASSERT(type) ((sizeof(type) <= NV_PLATFORM_MAX_IOCTL_SIZE) ? 1 : -1)
#define __NV_IOWR(nr, type) ({                                        \
    typedef char __NV_IOWR_TYPE_SIZE_ASSERT[__NV_IOWR_ASSERT(type)];  \
    _IOWR(NV_IOCTL_MAGIC, (nr), type);                                \
})

#define NV_PCI_DEV_FMT          "%04x:%02x:%02x.%x"
#define NV_PCI_DEV_FMT_ARGS(nv) (nv)->pci_info.domain, (nv)->pci_info.bus, \
                                (nv)->pci_info.slot, (nv)->pci_info.function

#define NV_RM_DEVICE_INTR_ADDRESS 0x100

/*!
 * @brief The order of the display clocks in the below defined enum
 * should be synced with below mapping array and macro.
 * All four should be updated simultaneously in case
 * of removal or addition of clocks in below order.
 * Also, TEGRA_DISP_WHICH_CLK_MAX is used in various places
 * in below mentioned files.
 * /chips_a/drivers/resman/arch/nvalloc/unix/Linux/nv-linux.h
 *
 * /chips_a/drivers/resman/arch/nvalloc/unix/src/os.c
 * dispClkMapRmToOsArr[] = {...};
 *
 * /chips_a/drivers/resman/arch/nvalloc/unix/Linux/nv-clk.c
 * osMapClk[] = {...};
 *
 */
typedef enum _TEGRA_DISP_WHICH_CLK
{
    TEGRA_DISP_WHICH_CLK_HUBCLK,
    TEGRA_DISP_WHICH_CLK_DISP,
    TEGRA_DISP_WHICH_CLK_P0,
    TEGRA_DISP_WHICH_CLK_P1,
    TEGRA_DISP_WHICH_CLK_MAX, // TEGRA_DISP_WHICH_CLK_MAX is defined for boundary checks only.
} TEGRA_DISP_WHICH_CLK;

#ifdef NVRM

extern const char *pNVRM_ID;

/*
 * ptr arithmetic convenience
 */

typedef union
{
    volatile NvV8 Reg008[1];
    volatile NvV16 Reg016[1];
    volatile NvV32 Reg032[1];
} nv_hwreg_t, * nv_phwreg_t;


#define NVRM_PCICFG_NUM_BARS            6
#define NVRM_PCICFG_BAR_OFFSET(i)       (0x10 + (i) * 4)
#define NVRM_PCICFG_BAR_REQTYPE_MASK    0x00000001
#define NVRM_PCICFG_BAR_REQTYPE_MEMORY  0x00000000
#define NVRM_PCICFG_BAR_MEMTYPE_MASK    0x00000006
#define NVRM_PCICFG_BAR_MEMTYPE_64BIT   0x00000004
#define NVRM_PCICFG_BAR_ADDR_MASK       0xfffffff0

#define NVRM_PCICFG_NUM_DWORDS          16

#define NV_GPU_NUM_BARS                 3
#define NV_GPU_BAR_INDEX_REGS           0
#define NV_GPU_BAR_INDEX_FB             1
#define NV_GPU_BAR_INDEX_IMEM           2

typedef struct
{
    NvU64 cpu_address;
    NvU64 strapped_size;
    NvU64 size;
    NvU32 offset;
    NvU32 *map;
    nv_phwreg_t map_u;
} nv_aperture_t;

typedef struct
{
    char *name;
    NvU32 *data;
} nv_parm_t;

#define NV_RM_PAGE_SHIFT    12
#define NV_RM_PAGE_SIZE     (1 << NV_RM_PAGE_SHIFT)
#define NV_RM_PAGE_MASK     (NV_RM_PAGE_SIZE - 1)

#define NV_RM_TO_OS_PAGE_SHIFT      (os_page_shift - NV_RM_PAGE_SHIFT)
#define NV_RM_PAGES_PER_OS_PAGE     (1U << NV_RM_TO_OS_PAGE_SHIFT)
#define NV_RM_PAGES_TO_OS_PAGES(count) \
    ((((NvUPtr)(count)) >> NV_RM_TO_OS_PAGE_SHIFT) + \
     ((((count) & ((1 << NV_RM_TO_OS_PAGE_SHIFT) - 1)) != 0) ? 1 : 0))

#if defined(NVCPU_X86_64)
#define NV_STACK_SIZE (NV_RM_PAGE_SIZE * 3)
#else
#define NV_STACK_SIZE (NV_RM_PAGE_SIZE * 2)
#endif

typedef struct nvidia_stack_s
{
    NvU32 size;
    void *top;
    NvU8  stack[NV_STACK_SIZE-16] __attribute__ ((aligned(16)));
} nvidia_stack_t;

/*
 * TODO: Remove once all UNIX layers have been converted to use nvidia_stack_t
 */
typedef nvidia_stack_t nv_stack_t;

typedef struct nv_file_private_t nv_file_private_t;

/*
 * this is a wrapper for unix events
 * unlike the events that will be returned to clients, this includes
 * kernel-specific data, such as file pointer, etc..
 */
typedef struct nv_event_s
{
    NvHandle            hParent;
    NvHandle            hObject;
    NvU32               index;
    NvU32               info32;
    NvU16               info16;
    nv_file_private_t  *nvfp;  /* per file-descriptor data pointer */
    NvU32               fd;
    NvBool              active; /* whether the event should be signaled */
    NvU32               refcount; /* count of associated RM events */
    struct nv_event_s  *next;
} nv_event_t;

typedef struct nv_kern_mapping_s
{
    void  *addr;
    NvU64 size;
    NvU32 modeFlag;
    struct nv_kern_mapping_s *next;
} nv_kern_mapping_t;

typedef struct nv_usermap_access_params_s
{
    NvU64    addr;
    NvU64    size;
    NvU64    offset;
    NvU64   *page_array;
    NvU64    num_pages;
    NvU64    mmap_start;
    NvU64    mmap_size;
    NvU64    access_start;
    NvU64    access_size;
    NvU64    remap_prot_extra;
    NvBool   contig;
} nv_usermap_access_params_t;

/*
 * It stores mapping context per mapping
 */
typedef struct nv_alloc_mapping_context_s {
    void  *alloc;
    NvU64  page_index;
    NvU64 *page_array;
    NvU64  num_pages;
    NvU64  mmap_start;
    NvU64  mmap_size;
    NvU64  access_start;
    NvU64  access_size;
    NvU64  remap_prot_extra;
    NvU32  prot;
    NvBool valid;
} nv_alloc_mapping_context_t;

typedef enum
{
    NV_SOC_IRQ_DISPLAY_TYPE,
    NV_SOC_IRQ_DPAUX_TYPE,
    NV_SOC_IRQ_GPIO_TYPE,
    NV_SOC_IRQ_HDACODEC_TYPE,
    NV_SOC_IRQ_INVALID_TYPE
} nv_soc_irq_type_t;

/*
 * It stores interrupt numbers and interrupt type and private data
 */
typedef struct nv_soc_irq_info_s {
    NvU32 irq_num;
    nv_soc_irq_type_t irq_type;
    NvBool bh_pending;
    union {
        NvU32 gpio_num;
        NvU32 dpaux_instance;
    } irq_data;
} nv_soc_irq_info_t;

#define NV_MAX_SOC_IRQS             6
#define NV_MAX_DPAUX_NUM_DEVICES    4

/*
 * per device state
 */

/* DMA-capable device data, defined by kernel interface layer */
typedef struct nv_dma_device nv_dma_device_t;

typedef struct nv_state_t
{
    void  *priv;                    /* private data */
    void  *os_state;                /* os-specific device state */

    int    flags;

    /* PCI config info */
    nv_pci_info_t pci_info;
    NvU16 subsystem_id;
    NvU16 subsystem_vendor;
    NvU32 gpu_id;
    NvU32 iovaspace_id;
    struct
    {
        NvBool         valid;
        NvU8           uuid[GPU_UUID_LEN];
    } nv_uuid_cache;
    void *handle;

    NvU32 pci_cfg_space[NVRM_PCICFG_NUM_DWORDS];

    /* physical characteristics */
    nv_aperture_t bars[NV_GPU_NUM_BARS];
    nv_aperture_t *regs;
    nv_aperture_t *dpaux[NV_MAX_DPAUX_NUM_DEVICES];
    nv_aperture_t *hdacodec_regs;
    nv_aperture_t *fb, ud;

    NvU32  num_dpaux_instance;
    NvU32  interrupt_line;
    NvU32  dpaux_irqs[NV_MAX_DPAUX_NUM_DEVICES];
    nv_soc_irq_info_t soc_irq_info[NV_MAX_SOC_IRQS];
    NvS32 current_soc_irq;
    NvU32 num_soc_irqs;
    NvU32 hdacodec_irq;

    NvBool primary_vga;

    NvU32 sim_env;

    NvU32 rc_timer_enabled;

    /* list of events allocated for this device */
    nv_event_t *event_list;

    /* lock to protect event_list */
    void *event_spinlock;

    nv_kern_mapping_t *kern_mappings;

    /* Kernel interface DMA device data */
    nv_dma_device_t *dma_dev;
    nv_dma_device_t *niso_dma_dev;

    /*
     * Per-GPU queue.  The actual queue object is usually allocated in the
     * arch-specific parent structure (e.g. nv_linux_state_t), and this
     * pointer just points to it.
     */
    struct os_work_queue *queue;

    /* Variable to denote if RM is running as firmware client or not */
    NvBool fw_client_rm;

    /* Variable to track, if nvidia_remove is called */
    NvBool removed;

    NvBool console_device;

    /* Variable to track, if GPU is external GPU */
    NvBool is_external_gpu;

    /* Variable to track, if regkey PreserveVideoMemoryAllocations is set */
    NvBool preserve_vidmem_allocations;

    /* Variable to force allocation of 32-bit addressable memory */
    NvBool force_dma32_alloc;

    /* Current cyclestats client and context */
    NvU32 profiler_owner;
    void *profiler_context;
} nv_state_t;

struct nv_file_private_t
{
    NvHandle *handles;
    NvU16 maxHandles;
    NvU32 deviceInstance;
    NvU8 metadata[64];
};

// Forward define the gpu ops structures
typedef struct gpuSession                           *nvgpuSessionHandle_t;
typedef struct gpuDevice                            *nvgpuDeviceHandle_t;
typedef struct gpuAddressSpace                      *nvgpuAddressSpaceHandle_t;
typedef struct gpuChannel                           *nvgpuChannelHandle_t;
typedef struct UvmGpuChannelInfo_tag                *nvgpuChannelInfo_t;
typedef struct UvmGpuChannelAllocParams_tag          nvgpuChannelAllocParams_t;
typedef struct UvmGpuCaps_tag                       *nvgpuCaps_t;
typedef struct UvmGpuCopyEnginesCaps_tag            *nvgpuCesCaps_t;
typedef struct UvmGpuAddressSpaceInfo_tag           *nvgpuAddressSpaceInfo_t;
typedef struct UvmGpuAllocInfo_tag                  *nvgpuAllocInfo_t;
typedef struct UvmGpuP2PCapsParams_tag              *nvgpuP2PCapsParams_t;
typedef struct gpuVaAllocInfo                       *nvgpuVaAllocInfo_t;
typedef struct gpuMapInfo                           *nvgpuMapInfo_t;
typedef struct UvmGpuFbInfo_tag                     *nvgpuFbInfo_t;
typedef struct UvmGpuEccInfo_tag                    *nvgpuEccInfo_t;
typedef struct UvmGpuFaultInfo_tag                  *nvgpuFaultInfo_t;
typedef struct UvmGpuAccessCntrInfo_tag             *nvgpuAccessCntrInfo_t;
typedef struct UvmGpuAccessCntrConfig_tag           *nvgpuAccessCntrConfig_t;
typedef struct UvmGpuInfo_tag                       nvgpuInfo_t;
typedef struct UvmGpuClientInfo_tag                 nvgpuClientInfo_t;
typedef struct gpuPmaAllocationOptions              *nvgpuPmaAllocationOptions_t;
typedef struct UvmPmaStatistics_tag                 *nvgpuPmaStatistics_t;
typedef struct UvmGpuMemoryInfo_tag                 *nvgpuMemoryInfo_t;
typedef struct UvmGpuExternalMappingInfo_tag        *nvgpuExternalMappingInfo_t;
typedef struct UvmGpuChannelResourceInfo_tag        *nvgpuChannelResourceInfo_t;
typedef struct UvmGpuChannelInstanceInfo_tag        *nvgpuChannelInstanceInfo_t;
typedef struct UvmGpuChannelResourceBindParams_tag  *nvgpuChannelResourceBindParams_t;
typedef NV_STATUS (*nvPmaEvictPagesCallback)(void *, NvU32, NvU64 *, NvU32, NvU64, NvU64);
typedef NV_STATUS (*nvPmaEvictRangeCallback)(void *, NvU64, NvU64);

/*
 * flags
 */

#define NV_FLAG_OPEN                   0x0001
#define NV_FLAG_EXCLUDE                0x0002
#define NV_FLAG_CONTROL                0x0004
// Unused                              0x0008
#define NV_FLAG_SOC_DISPLAY            0x0010
#define NV_FLAG_USES_MSI               0x0020
#define NV_FLAG_USES_MSIX              0x0040
#define NV_FLAG_PASSTHRU               0x0080
#define NV_FLAG_SUSPENDED              0x0100
// Unused                              0x0200
// Unused                              0x0400
#define NV_FLAG_PERSISTENT_SW_STATE    0x0800
#define NV_FLAG_IN_RECOVERY            0x1000
// Unused                              0x2000
#define NV_FLAG_UNBIND_LOCK            0x4000
/* To be set when GPU is not present on the bus, to help device teardown */
#define NV_FLAG_IN_SURPRISE_REMOVAL    0x8000

typedef enum
{
    NV_PM_ACTION_HIBERNATE,
    NV_PM_ACTION_STANDBY,
    NV_PM_ACTION_RESUME
} nv_pm_action_t;

typedef enum
{
    NV_PM_ACTION_DEPTH_DEFAULT,
    NV_PM_ACTION_DEPTH_MODESET,
    NV_PM_ACTION_DEPTH_UVM
} nv_pm_action_depth_t;

typedef enum
{
    NV_DYNAMIC_PM_NEVER,
    NV_DYNAMIC_PM_COARSE,
    NV_DYNAMIC_PM_FINE
} nv_dynamic_power_mode_t;

typedef enum
{
    NV_POWER_STATE_IN_HIBERNATE,
    NV_POWER_STATE_IN_STANDBY,
    NV_POWER_STATE_RUNNING
} nv_power_state_t;

#define NV_PRIMARY_VGA(nv)      ((nv)->primary_vga)

#define NV_IS_CTL_DEVICE(nv)    ((nv)->flags & NV_FLAG_CONTROL)
#define NV_IS_SOC_DISPLAY_DEVICE(nv)    \
        ((nv)->flags & NV_FLAG_SOC_DISPLAY)

#define NV_IS_DEVICE_IN_SURPRISE_REMOVAL(nv)    \
        (((nv)->flags & NV_FLAG_IN_SURPRISE_REMOVAL) != 0)

/*
 * NVIDIA ACPI event IDs to be passed into the core NVIDIA
 * driver for various events like display switch events,
 * AC/battery events, docking events, etc..
 */
#define NV_SYSTEM_ACPI_DISPLAY_SWITCH_EVENT  0x8001
#define NV_SYSTEM_ACPI_BATTERY_POWER_EVENT   0x8002
#define NV_SYSTEM_ACPI_DOCK_EVENT            0x8003

/*
 * GPU add/remove events
 */
#define NV_SYSTEM_GPU_ADD_EVENT             0x9001
#define NV_SYSTEM_GPU_REMOVE_EVENT          0x9002

/*
 * Status bit definitions for display switch hotkey events.
 */
#define NV_HOTKEY_STATUS_DISPLAY_ENABLE_LCD 0x01
#define NV_HOTKEY_STATUS_DISPLAY_ENABLE_CRT 0x02
#define NV_HOTKEY_STATUS_DISPLAY_ENABLE_TV  0x04
#define NV_HOTKEY_STATUS_DISPLAY_ENABLE_DFP 0x08

/*
 * NVIDIA ACPI sub-event IDs (event types) to be passed into
 * to core NVIDIA driver for ACPI events.
 */
#define NV_SYSTEM_ACPI_EVENT_VALUE_DISPLAY_SWITCH_DEFAULT    0
#define NV_SYSTEM_ACPI_EVENT_VALUE_POWER_EVENT_AC            0
#define NV_SYSTEM_ACPI_EVENT_VALUE_POWER_EVENT_BATTERY       1
#define NV_SYSTEM_ACPI_EVENT_VALUE_DOCK_EVENT_UNDOCKED       0
#define NV_SYSTEM_ACPI_EVENT_VALUE_DOCK_EVENT_DOCKED         1

#define NV_ACPI_NVIF_HANDLE_PRESENT 0x01
#define NV_ACPI_DSM_HANDLE_PRESENT  0x02
#define NV_ACPI_WMMX_HANDLE_PRESENT 0x04
#define NV_ACPI_MXMI_HANDLE_PRESENT 0x08
#define NV_ACPI_MXMS_HANDLE_PRESENT 0x10

#define NV_EVAL_ACPI_METHOD_NVIF     0x01
#define NV_EVAL_ACPI_METHOD_WMMX     0x02
#define NV_EVAL_ACPI_METHOD_MXMI     0x03
#define NV_EVAL_ACPI_METHOD_MXMS     0x04

#define NV_I2C_CMD_READ              1
#define NV_I2C_CMD_WRITE             2
#define NV_I2C_CMD_SMBUS_READ        3
#define NV_I2C_CMD_SMBUS_WRITE       4
#define NV_I2C_CMD_SMBUS_QUICK_WRITE 5
#define NV_I2C_CMD_SMBUS_QUICK_READ  6
#define NV_I2C_CMD_SMBUS_BLOCK_READ  7
#define NV_I2C_CMD_SMBUS_BLOCK_WRITE 8

// Flags needed by OSAllocPagesNode
#define NV_ALLOC_PAGES_NODE_NONE                0x0
#define NV_ALLOC_PAGES_NODE_SCRUB_ON_ALLOC      0x1
#define NV_ALLOC_PAGES_NODE_FORCE_ALLOC         0x2

/*
** where we hide our nv_state_t * ...
*/
#define NV_SET_NV_STATE(pgpu,p) ((pgpu)->pOsGpuInfo = (p))
#define NV_GET_NV_STATE(pGpu) \
    (nv_state_t *)((pGpu) ? (pGpu)->pOsGpuInfo : NULL)

#define IS_REG_OFFSET(nv, offset, length)                                       \
    (((offset) >= (nv)->regs->cpu_address) &&                                   \
    (((offset) + ((length)-1)) <=                                               \
        (nv)->regs->cpu_address + ((nv)->regs->size-1)))

#define IS_FB_OFFSET(nv, offset, length)                                        \
    (((nv)->fb) && ((offset) >= (nv)->fb->cpu_address) &&                       \
    (((offset) + ((length)-1)) <= (nv)->fb->cpu_address + ((nv)->fb->size-1)))

#define IS_UD_OFFSET(nv, offset, length)                                        \
    (((nv)->ud.cpu_address != 0) && ((nv)->ud.size != 0) &&                     \
    ((offset) >= (nv)->ud.cpu_address) &&                                       \
    (((offset) + ((length)-1)) <= (nv)->ud.cpu_address + ((nv)->ud.size-1)))

#define IS_IMEM_OFFSET(nv, offset, length)                                      \
    (((nv)->bars[NV_GPU_BAR_INDEX_IMEM].cpu_address != 0) &&                    \
     ((nv)->bars[NV_GPU_BAR_INDEX_IMEM].size != 0) &&                           \
     ((offset) >= (nv)->bars[NV_GPU_BAR_INDEX_IMEM].cpu_address) &&             \
     (((offset) + ((length) - 1)) <=                                            \
        (nv)->bars[NV_GPU_BAR_INDEX_IMEM].cpu_address +                         \
            ((nv)->bars[NV_GPU_BAR_INDEX_IMEM].size - 1)))

#define NV_RM_MAX_MSIX_LINES  8

#define NV_MAX_ISR_DELAY_US           20000
#define NV_MAX_ISR_DELAY_MS           (NV_MAX_ISR_DELAY_US / 1000)

#define NV_TIMERCMP(a, b, CMP)                                              \
    (((a)->tv_sec == (b)->tv_sec) ?                                         \
        ((a)->tv_usec CMP (b)->tv_usec) : ((a)->tv_sec CMP (b)->tv_sec))

#define NV_TIMERADD(a, b, result)                                           \
    {                                                                       \
        (result)->tv_sec = (a)->tv_sec + (b)->tv_sec;                       \
        (result)->tv_usec = (a)->tv_usec + (b)->tv_usec;                    \
        if ((result)->tv_usec >= 1000000)                                   \
        {                                                                   \
            ++(result)->tv_sec;                                             \
            (result)->tv_usec -= 1000000;                                   \
        }                                                                   \
    }

#define NV_TIMERSUB(a, b, result)                                           \
    {                                                                       \
        (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;                       \
        (result)->tv_usec = (a)->tv_usec - (b)->tv_usec;                    \
        if ((result)->tv_usec < 0)                                          \
        {                                                                   \
          --(result)->tv_sec;                                               \
          (result)->tv_usec += 1000000;                                     \
        }                                                                   \
    }

#define NV_TIMEVAL_TO_US(tv)    ((NvU64)(tv).tv_sec * 1000000 + (tv).tv_usec)

#ifndef NV_ALIGN_UP
#define NV_ALIGN_UP(v,g) (((v) + ((g) - 1)) & ~((g) - 1))
#endif
#ifndef NV_ALIGN_DOWN
#define NV_ALIGN_DOWN(v,g) ((v) & ~((g) - 1))
#endif

/*
 * driver internal interfaces
 */

/*
 * ---------------------------------------------------------------------------
 *
 * Function prototypes for UNIX specific OS interface.
 *
 * ---------------------------------------------------------------------------
 */

NvU32      NV_API_CALL  nv_get_dev_minor         (nv_state_t *);
void*      NV_API_CALL  nv_alloc_kernel_mapping  (nv_state_t *, void *, NvU64, NvU32, NvU64, void **);
NV_STATUS  NV_API_CALL  nv_free_kernel_mapping   (nv_state_t *, void *, void *, void *);
NV_STATUS  NV_API_CALL  nv_alloc_user_mapping    (nv_state_t *, void *, NvU64, NvU32, NvU64, NvU32, NvU64 *, void **);
NV_STATUS  NV_API_CALL  nv_free_user_mapping     (nv_state_t *, void *, NvU64, void *);
NV_STATUS  NV_API_CALL  nv_add_mapping_context_to_file (nv_state_t *, nv_usermap_access_params_t*, NvU32, void *, NvU64, NvU32);

NvU64  NV_API_CALL  nv_get_kern_phys_address     (NvU64);
NvU64  NV_API_CALL  nv_get_user_phys_address     (NvU64);
nv_state_t*  NV_API_CALL  nv_get_adapter_state   (NvU32, NvU8, NvU8);
nv_state_t*  NV_API_CALL  nv_get_ctl_state       (void);

void   NV_API_CALL  nv_set_dma_address_size      (nv_state_t *, NvU32 );

NV_STATUS  NV_API_CALL  nv_alias_pages           (nv_state_t *, NvU32, NvU32, NvU32, NvU64, NvU64 *, void **);
NV_STATUS  NV_API_CALL  nv_alloc_pages           (nv_state_t *, NvU32, NvBool, NvU32, NvBool, NvU64 *, void **);
NV_STATUS  NV_API_CALL  nv_free_pages            (nv_state_t *, NvU32, NvBool, NvU32, void *);

NV_STATUS  NV_API_CALL  nv_register_user_pages   (nv_state_t *, NvU64, NvU64 *, void *, void **);
void       NV_API_CALL  nv_unregister_user_pages (nv_state_t *, NvU64, void **, void **);

NV_STATUS NV_API_CALL   nv_register_peer_io_mem  (nv_state_t *, NvU64 *, NvU64, void **);
void      NV_API_CALL   nv_unregister_peer_io_mem(nv_state_t *, void *);

NV_STATUS NV_API_CALL   nv_register_phys_pages   (nv_state_t *, NvU64 *, NvU64, NvU32, void **);
void      NV_API_CALL   nv_unregister_phys_pages (nv_state_t *, void *);

NV_STATUS  NV_API_CALL  nv_dma_map_pages         (nv_dma_device_t *, NvU64, NvU64 *, NvBool, NvU32, void **);
NV_STATUS  NV_API_CALL  nv_dma_unmap_pages       (nv_dma_device_t *, NvU64, NvU64 *, void **);

NV_STATUS  NV_API_CALL  nv_dma_map_alloc         (nv_dma_device_t *, NvU64, NvU64 *, NvBool, void **);
NV_STATUS  NV_API_CALL  nv_dma_unmap_alloc       (nv_dma_device_t *, NvU64, NvU64 *, void **);

NV_STATUS  NV_API_CALL  nv_dma_map_peer          (nv_dma_device_t *, nv_dma_device_t *, NvU8, NvU64, NvU64 *);
void       NV_API_CALL  nv_dma_unmap_peer        (nv_dma_device_t *, NvU64, NvU64);

NV_STATUS  NV_API_CALL  nv_dma_map_mmio          (nv_dma_device_t *, NvU64, NvU64 *);
void       NV_API_CALL  nv_dma_unmap_mmio        (nv_dma_device_t *, NvU64, NvU64);

void       NV_API_CALL  nv_dma_cache_invalidate  (nv_dma_device_t *, void *);
void       NV_API_CALL  nv_dma_enable_nvlink     (nv_dma_device_t *);

NvS32  NV_API_CALL  nv_start_rc_timer            (nv_state_t *);
NvS32  NV_API_CALL  nv_stop_rc_timer             (nv_state_t *);

void   NV_API_CALL  nv_post_event                (nv_event_t *, NvHandle, NvU32, NvU32, NvU16, NvBool);
NvS32  NV_API_CALL  nv_get_event                 (nv_file_private_t *, nv_event_t *, NvU32 *);

void*  NV_API_CALL  nv_i2c_add_adapter           (nv_state_t *, NvU32);
void   NV_API_CALL  nv_i2c_del_adapter           (nv_state_t *, void *);

void   NV_API_CALL  nv_acpi_methods_init         (NvU32 *);
void   NV_API_CALL  nv_acpi_methods_uninit       (void);

NV_STATUS  NV_API_CALL  nv_acpi_method           (NvU32, NvU32, NvU32, void *, NvU16, NvU32 *, void *, NvU16 *);
NV_STATUS  NV_API_CALL  nv_acpi_dsm_method       (nv_state_t *, NvU8 *, NvU32, NvU32, void *, NvU16, NvU32 *, void *, NvU16 *);
NV_STATUS  NV_API_CALL  nv_acpi_ddc_method       (nv_state_t *, void *, NvU32 *, NvBool);
NV_STATUS  NV_API_CALL  nv_acpi_dod_method       (nv_state_t *, NvU32 *, NvU32 *);
NV_STATUS  NV_API_CALL  nv_acpi_rom_method       (nv_state_t *, NvU32 *, NvU32 *);
NV_STATUS  NV_API_CALL  nv_acpi_get_powersource  (NvU32 *);

NV_STATUS  NV_API_CALL  nv_acpi_mux_method       (nv_state_t *, NvU32 *, NvU32, char *);

NV_STATUS  NV_API_CALL  nv_log_error             (nv_state_t *, NvU32, const char *, va_list);

NvU64      NV_API_CALL  nv_get_dma_start_address (nv_state_t *);
NV_STATUS  NV_API_CALL  nv_set_primary_vga_status(nv_state_t *);
NV_STATUS  NV_API_CALL  nv_pci_trigger_recovery  (nv_state_t *);
NvBool     NV_API_CALL  nv_requires_dma_remap    (nv_state_t *);

nv_file_private_t* NV_API_CALL nv_get_file_private(NvS32, NvBool, void **);
void               NV_API_CALL nv_put_file_private(void *);

NV_STATUS NV_API_CALL nv_get_device_memory_config(nv_state_t *, NvU32 *, NvU32 *, NvU32 *, NvU32 *, NvS32 *);

NV_STATUS NV_API_CALL nv_get_ibmnpu_genreg_info(nv_state_t *, NvU64 *, NvU64 *, void**);
NV_STATUS NV_API_CALL nv_get_ibmnpu_relaxed_ordering_mode(nv_state_t *nv, NvBool *mode);

void      NV_API_CALL nv_wait_for_ibmnpu_rsync(nv_state_t *nv);

void      NV_API_CALL nv_ibmnpu_cache_flush_range(nv_state_t *nv, NvU64, NvU64);

void      NV_API_CALL nv_p2p_free_platform_data(void *data);

#if defined(NVCPU_PPC64LE)
NV_STATUS NV_API_CALL nv_get_nvlink_line_rate    (nv_state_t *, NvU32 *);
#endif

void      NV_API_CALL nv_register_backlight      (nv_state_t *, NvU32, NvU32);
void      NV_API_CALL nv_unregister_backlight    (nv_state_t *);

NV_STATUS NV_API_CALL nv_revoke_gpu_mappings     (nv_state_t *);
void      NV_API_CALL nv_acquire_mmap_lock       (nv_state_t *);
void      NV_API_CALL nv_release_mmap_lock       (nv_state_t *);
NvBool    NV_API_CALL nv_get_all_mappings_revoked_locked (nv_state_t *);
void      NV_API_CALL nv_set_safe_to_mmap_locked (nv_state_t *, NvBool);

NV_STATUS NV_API_CALL nv_indicate_idle           (nv_state_t *);
NV_STATUS NV_API_CALL nv_indicate_not_idle       (nv_state_t *);
void      NV_API_CALL nv_idle_holdoff            (nv_state_t *);

NvBool    NV_API_CALL nv_dynamic_power_available (nv_state_t *);
void      NV_API_CALL nv_audio_dynamic_power     (nv_state_t *);

void      NV_API_CALL nv_control_soc_irqs        (nv_state_t *, NvBool bEnable);
NV_STATUS NV_API_CALL nv_get_current_irq_priv_data(nv_state_t *, NvU32 *);

NV_STATUS NV_API_CALL nv_acquire_fabric_mgmt_cap (int, int*);
int       NV_API_CALL nv_cap_drv_init(void);
void      NV_API_CALL nv_cap_drv_exit(void);

struct dma_buf;
typedef struct nv_dma_buf nv_dma_buf_t;

NV_STATUS NV_API_CALL nv_dma_import_dma_buf      (nv_dma_device_t *, struct dma_buf *, NvU32 *, void **, nv_dma_buf_t **);
NV_STATUS NV_API_CALL nv_dma_import_from_fd      (nv_dma_device_t *, NvS32, NvU32 *, void **, nv_dma_buf_t **);
void      NV_API_CALL nv_dma_release_dma_buf     (void *, nv_dma_buf_t *);

void      NV_API_CALL nv_schedule_uvm_isr        (nv_state_t *);


NvBool    NV_API_CALL nv_platform_supports_s0ix  (void);
NvBool    NV_API_CALL nv_s2idle_pm_configured    (void);




































/*
 * ---------------------------------------------------------------------------
 *
 * Function prototypes for Resource Manager interface.
 *
 * ---------------------------------------------------------------------------
 */

NvBool     NV_API_CALL  rm_init_rm               (nvidia_stack_t *);
void       NV_API_CALL  rm_shutdown_rm           (nvidia_stack_t *);
NvBool     NV_API_CALL  rm_init_private_state    (nvidia_stack_t *, nv_state_t *);
void       NV_API_CALL  rm_free_private_state    (nvidia_stack_t *, nv_state_t *);
NvBool     NV_API_CALL  rm_init_adapter          (nvidia_stack_t *, nv_state_t *);
void       NV_API_CALL  rm_disable_adapter       (nvidia_stack_t *, nv_state_t *);
void       NV_API_CALL  rm_shutdown_adapter      (nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL  rm_exclude_adapter       (nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL  rm_acquire_api_lock      (nvidia_stack_t *);
NV_STATUS  NV_API_CALL  rm_release_api_lock      (nvidia_stack_t *);
NV_STATUS  NV_API_CALL  rm_ioctl                 (nvidia_stack_t *, nv_state_t *, nv_file_private_t *, NvU32, void *, NvU32);
NvBool     NV_API_CALL  rm_isr                   (nvidia_stack_t *, nv_state_t *, NvU32 *);
void       NV_API_CALL  rm_isr_bh                (nvidia_stack_t *, nv_state_t *);
void       NV_API_CALL  rm_isr_bh_unlocked       (nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL  rm_power_management      (nvidia_stack_t *, nv_state_t *, nv_pm_action_t);
NV_STATUS  NV_API_CALL  rm_stop_user_channels    (nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL  rm_restart_user_channels (nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL  rm_save_low_res_mode     (nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL  rm_get_vbios_version     (nvidia_stack_t *, nv_state_t *, NvU32 *, NvU32 *, NvU32 *, NvU32 *, NvU32 *);
char*      NV_API_CALL  rm_get_gpu_uuid          (nvidia_stack_t *, nv_state_t *);
const NvU8* NV_API_CALL rm_get_gpu_uuid_raw      (nvidia_stack_t *, nv_state_t *);
void       NV_API_CALL  rm_cleanup_file_private  (nvidia_stack_t *, nv_state_t *, nv_file_private_t *);
void       NV_API_CALL  rm_unbind_lock           (nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL  rm_read_registry_dword   (nvidia_stack_t *, nv_state_t *, const char *, NvU32 *);
NV_STATUS  NV_API_CALL  rm_write_registry_dword  (nvidia_stack_t *, nv_state_t *, const char *, NvU32);
NV_STATUS  NV_API_CALL  rm_write_registry_binary (nvidia_stack_t *, nv_state_t *, const char *, NvU8 *, NvU32);
NV_STATUS  NV_API_CALL  rm_write_registry_string (nvidia_stack_t *, nv_state_t *, const char *, const char *, NvU32);
void       NV_API_CALL  rm_parse_option_string   (nvidia_stack_t *, const char *);
char*      NV_API_CALL  rm_remove_spaces         (const char *);
char*      NV_API_CALL  rm_string_token          (char **, const char);

NV_STATUS  NV_API_CALL  rm_run_rc_callback       (nvidia_stack_t *, nv_state_t *);
void       NV_API_CALL  rm_execute_work_item     (nvidia_stack_t *, void *);
const char* NV_API_CALL rm_get_device_name       (NvU16, NvU16, NvU16);

NvU64      NV_API_CALL  nv_rdtsc                 (void);

NV_STATUS  NV_API_CALL  rm_is_supported_device   (nvidia_stack_t *, nv_state_t *);
NvBool     NV_API_CALL  rm_is_supported_pci_device(NvU8   pci_class,
                                                   NvU8   pci_subclass,
                                                   NvU16  vendor,
                                                   NvU16  device,
                                                   NvU16  subsystem_vendor,
                                                   NvU16  subsystem_device,
                                                   NvBool print_legacy_warning);

void       NV_API_CALL  rm_i2c_remove_adapters    (nvidia_stack_t *, nv_state_t *);
NvBool     NV_API_CALL  rm_i2c_is_smbus_capable   (nvidia_stack_t *, nv_state_t *, void *);
NV_STATUS  NV_API_CALL  rm_i2c_transfer           (nvidia_stack_t *, nv_state_t *, void *, NvU8, NvU8, NvU8, NvU32, NvU8 *);

NV_STATUS  NV_API_CALL  rm_perform_version_check  (nvidia_stack_t *, void *, NvU32);

NV_STATUS  NV_API_CALL  rm_system_event           (nvidia_stack_t *, NvU32, NvU32);

void       NV_API_CALL  rm_disable_gpu_state_persistence    (nvidia_stack_t *sp, nv_state_t *);
NV_STATUS  NV_API_CALL  rm_p2p_init_mapping       (nvidia_stack_t *, NvU64, NvU64 *, NvU64 *, NvU64 *, NvU64 *, NvU64, NvU64, NvU64, NvU64, void (*)(void *), void *);
NV_STATUS  NV_API_CALL  rm_p2p_destroy_mapping    (nvidia_stack_t *, NvU64);
NV_STATUS  NV_API_CALL  rm_p2p_get_pages          (nvidia_stack_t *, NvU64, NvU32, NvU64, NvU64, NvU64 *, NvU32 *, NvU32 *, NvU32 *, NvU8 **, void *);
NV_STATUS  NV_API_CALL  rm_p2p_register_callback  (nvidia_stack_t *, NvU64, NvU64, NvU64, void *, void (*)(void *), void *);
NV_STATUS  NV_API_CALL  rm_p2p_put_pages          (nvidia_stack_t *, NvU64, NvU32, NvU64, void *);
NV_STATUS  NV_API_CALL  rm_p2p_dma_map_pages      (nvidia_stack_t *, nv_dma_device_t *, NvU8 *, NvU32, NvU32, NvU64 *, void **);
NV_STATUS  NV_API_CALL  rm_log_gpu_crash          (nv_stack_t *, nv_state_t *);

NV_STATUS  NV_API_CALL  rm_gpu_ops_create_session (nvidia_stack_t *, nvgpuSessionHandle_t *);
NV_STATUS  NV_API_CALL  rm_gpu_ops_destroy_session (nvidia_stack_t *, nvgpuSessionHandle_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_device_create (nvidia_stack_t *, nvgpuSessionHandle_t, const nvgpuInfo_t *, const NvProcessorUuid *, nvgpuDeviceHandle_t *);
NV_STATUS  NV_API_CALL  rm_gpu_ops_device_destroy (nvidia_stack_t *, nvgpuDeviceHandle_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_address_space_create(nvidia_stack_t *, nvgpuDeviceHandle_t, unsigned long long, unsigned long long, nvgpuAddressSpaceHandle_t *, nvgpuAddressSpaceInfo_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_dup_address_space(nvidia_stack_t *, nvgpuDeviceHandle_t, NvHandle, NvHandle, nvgpuAddressSpaceHandle_t *, nvgpuAddressSpaceInfo_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_address_space_destroy(nvidia_stack_t *, nvgpuAddressSpaceHandle_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_memory_alloc_fb(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvLength, NvU64 *, nvgpuAllocInfo_t);

NV_STATUS  NV_API_CALL  rm_gpu_ops_pma_alloc_pages(nvidia_stack_t *, void *, NvLength, NvU32 , nvgpuPmaAllocationOptions_t, NvU64 *);
NV_STATUS  NV_API_CALL  rm_gpu_ops_pma_free_pages(nvidia_stack_t *, void *, NvU64 *, NvLength , NvU32, NvU32);
NV_STATUS  NV_API_CALL  rm_gpu_ops_pma_pin_pages(nvidia_stack_t *, void *, NvU64 *, NvLength , NvU32, NvU32);
NV_STATUS  NV_API_CALL  rm_gpu_ops_pma_unpin_pages(nvidia_stack_t *, void *, NvU64 *, NvLength , NvU32);
NV_STATUS  NV_API_CALL  rm_gpu_ops_get_pma_object(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, void **, const nvgpuPmaStatistics_t *);
NV_STATUS  NV_API_CALL  rm_gpu_ops_pma_register_callbacks(nvidia_stack_t *sp, void *, nvPmaEvictPagesCallback, nvPmaEvictRangeCallback, void *);
void       NV_API_CALL  rm_gpu_ops_pma_unregister_callbacks(nvidia_stack_t *sp, void *);

NV_STATUS  NV_API_CALL  rm_gpu_ops_memory_alloc_sys(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvLength, NvU64 *, nvgpuAllocInfo_t);

NV_STATUS  NV_API_CALL  rm_gpu_ops_get_p2p_caps(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuAddressSpaceHandle_t, nvgpuP2PCapsParams_t);

NV_STATUS  NV_API_CALL  rm_gpu_ops_memory_cpu_map(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvU64, NvLength, void **, NvU32);
NV_STATUS  NV_API_CALL  rm_gpu_ops_memory_cpu_ummap(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, void*);
NV_STATUS  NV_API_CALL  rm_gpu_ops_channel_allocate(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, const nvgpuChannelAllocParams_t *, nvgpuChannelHandle_t *, nvgpuChannelInfo_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_channel_destroy(nvidia_stack_t *, nvgpuChannelHandle_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_memory_free(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvU64);
NV_STATUS  NV_API_CALL rm_gpu_ops_query_caps(nvidia_stack_t *, nvgpuDeviceHandle_t, nvgpuCaps_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_query_ces_caps(nvidia_stack_t *sp, nvgpuAddressSpaceHandle_t, nvgpuCesCaps_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_gpu_info(nvidia_stack_t *, const NvProcessorUuid *pUuid, const nvgpuClientInfo_t *, nvgpuInfo_t *);
NV_STATUS  NV_API_CALL rm_gpu_ops_service_device_interrupts_rm(nvidia_stack_t *, nvgpuDeviceHandle_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_dup_allocation(nvidia_stack_t *, NvHandle, nvgpuAddressSpaceHandle_t, NvU64, nvgpuAddressSpaceHandle_t, NvU64*, NvBool);

NV_STATUS  NV_API_CALL  rm_gpu_ops_dup_memory (nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvHandle, NvHandle, NvHandle *, nvgpuMemoryInfo_t);

NV_STATUS  NV_API_CALL rm_gpu_ops_free_duped_handle(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvHandle);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_fb_info(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuFbInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_ecc_info(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuEccInfo_t);
NV_STATUS NV_API_CALL rm_gpu_ops_own_page_fault_intr(nvidia_stack_t *, nvgpuDeviceHandle_t, NvBool);
NV_STATUS  NV_API_CALL rm_gpu_ops_init_fault_info(nvidia_stack_t *, nvgpuDeviceHandle_t, nvgpuFaultInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_destroy_fault_info(nvidia_stack_t *, nvgpuDeviceHandle_t, nvgpuFaultInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_non_replayable_faults(nvidia_stack_t *, nvgpuFaultInfo_t, void *, NvU32 *);
NV_STATUS  NV_API_CALL rm_gpu_ops_has_pending_non_replayable_faults(nvidia_stack_t *, nvgpuFaultInfo_t, NvBool *);
NV_STATUS  NV_API_CALL rm_gpu_ops_init_access_cntr_info(nvidia_stack_t *, nvgpuDeviceHandle_t, nvgpuAccessCntrInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_destroy_access_cntr_info(nvidia_stack_t *, nvgpuDeviceHandle_t, nvgpuAccessCntrInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_own_access_cntr_intr(nvidia_stack_t *, nvgpuSessionHandle_t, nvgpuAccessCntrInfo_t, NvBool);
NV_STATUS  NV_API_CALL rm_gpu_ops_enable_access_cntr(nvidia_stack_t *, nvgpuDeviceHandle_t, nvgpuAccessCntrInfo_t, nvgpuAccessCntrConfig_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_disable_access_cntr(nvidia_stack_t *, nvgpuDeviceHandle_t, nvgpuAccessCntrInfo_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_set_page_directory (nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvU64, unsigned, NvBool);
NV_STATUS  NV_API_CALL  rm_gpu_ops_unset_page_directory (nvidia_stack_t *, nvgpuAddressSpaceHandle_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_p2p_object_create(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuAddressSpaceHandle_t, NvHandle *);
void       NV_API_CALL rm_gpu_ops_p2p_object_destroy(nvidia_stack_t *, nvgpuSessionHandle_t, NvHandle);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_external_alloc_ptes(nvidia_stack_t*, nvgpuAddressSpaceHandle_t, NvHandle, NvU64, NvU64, nvgpuExternalMappingInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_retain_channel(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvHandle, NvHandle, void **, nvgpuChannelInstanceInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_retain_channel_resources(nvidia_stack_t *, void *, nvgpuChannelResourceInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_bind_channel_resources(nvidia_stack_t *, void *, nvgpuChannelResourceBindParams_t);
void       NV_API_CALL rm_gpu_ops_release_channel(nvidia_stack_t *, void *);
void       NV_API_CALL rm_gpu_ops_release_channel_resources(nvidia_stack_t *, NvP64*, NvU32);
void       NV_API_CALL rm_gpu_ops_stop_channel(nvidia_stack_t *, void *, NvBool);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_channel_resource_ptes(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvP64, NvU64, NvU64, nvgpuExternalMappingInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_report_non_replayable_fault(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, const void *);
void       NV_API_CALL rm_kernel_rmapi_op(nvidia_stack_t *sp, void *ops_cmd);
NvBool     NV_API_CALL rm_get_device_remove_flag(nvidia_stack_t *sp, NvU32 gpu_id);
NV_STATUS  NV_API_CALL rm_gpu_copy_mmu_faults(nvidia_stack_t *, nv_state_t *, NvU32 *);
NV_STATUS  NV_API_CALL rm_gpu_copy_mmu_faults_unlocked(nvidia_stack_t *, nv_state_t *, NvU32 *);
NV_STATUS  NV_API_CALL rm_gpu_need_4k_page_isolation(nv_state_t *, NvBool *);
NvBool     NV_API_CALL rm_is_chipset_io_coherent(nv_stack_t *);
NvBool     NV_API_CALL rm_init_event_locks(nvidia_stack_t *, nv_state_t *);
void       NV_API_CALL rm_destroy_event_locks(nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL rm_get_gpu_numa_info(nvidia_stack_t *, nv_state_t *, NvS32 *, NvU64 *, NvU64 *, NvU64 *, NvU32 *);
NV_STATUS  NV_API_CALL rm_set_backlight(nvidia_stack_t *, nv_state_t *, NvU32, NvU32);
NV_STATUS  NV_API_CALL rm_get_backlight(nvidia_stack_t *, nv_state_t *, NvU32, NvU32 *);
NV_STATUS  NV_API_CALL rm_gpu_numa_online(nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL rm_gpu_numa_offline(nvidia_stack_t *, nv_state_t *);
NvBool     NV_API_CALL rm_is_device_sequestered(nvidia_stack_t *, nv_state_t *);
void       NV_API_CALL rm_check_for_gpu_surprise_removal(nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL rm_set_external_kernel_client_count(nvidia_stack_t *, nv_state_t *, NvBool);
NV_STATUS  NV_API_CALL rm_schedule_gpu_wakeup(nvidia_stack_t *, nv_state_t *);
NvBool     NV_API_CALL rm_is_iommu_needed_for_sriov(nvidia_stack_t *, nv_state_t *);

void       NV_API_CALL rm_init_dynamic_power_management(nvidia_stack_t *, nv_state_t *, NvBool);
void       NV_API_CALL rm_cleanup_dynamic_power_management(nvidia_stack_t *, nv_state_t *);
NV_STATUS  NV_API_CALL rm_ref_dynamic_power(nvidia_stack_t *, nv_state_t *, nv_dynamic_power_mode_t);
void       NV_API_CALL rm_unref_dynamic_power(nvidia_stack_t *, nv_state_t *, nv_dynamic_power_mode_t);
NV_STATUS  NV_API_CALL rm_transition_dynamic_power(nvidia_stack_t *, nv_state_t *, NvBool);
const char* NV_API_CALL rm_get_vidmem_power_status(nvidia_stack_t *, nv_state_t *);
const char* NV_API_CALL rm_get_dynamic_power_management_status(nvidia_stack_t *, nv_state_t *);
const char* NV_API_CALL rm_get_gpu_gcx_support(nvidia_stack_t *, nv_state_t *, NvBool);

void       NV_API_CALL rm_acpi_notify(nvidia_stack_t *, nv_state_t *, NvU32);

/* vGPU VFIO specific functions */
NV_STATUS  NV_API_CALL  nv_vgpu_create_request(nvidia_stack_t *, nv_state_t *, const NvU8 *, NvU32, NvU16 *, NvU32);
NV_STATUS  NV_API_CALL  nv_vgpu_delete(nvidia_stack_t *, const NvU8 *, NvU16);
NV_STATUS  NV_API_CALL  nv_vgpu_get_type_ids(nvidia_stack_t *, nv_state_t *, NvU32 *, NvU32 **, NvBool);
NV_STATUS  NV_API_CALL  nv_vgpu_get_type_info(nvidia_stack_t *, nv_state_t *, NvU32, char *, int, NvU8);
NV_STATUS  NV_API_CALL  nv_vgpu_get_bar_info(nvidia_stack_t *, nv_state_t *, const NvU8 *, NvU64 *, NvU32, void *, void **);
NV_STATUS  NV_API_CALL  nv_vgpu_start(nvidia_stack_t *, const NvU8 *, void *, NvS32 *, NvU8 *, NvU32);
NV_STATUS  NV_API_CALL  nv_vgpu_get_sparse_mmap(nvidia_stack_t *, nv_state_t *, const NvU8 *, NvU64 **, NvU64 **, NvU32 *);
NV_STATUS  NV_API_CALL  nv_vgpu_process_vf_info(nvidia_stack_t *, nv_state_t *, NvU8, NvU32, NvU8, NvU8, NvU8, NvBool, void *);
NV_STATUS  NV_API_CALL  nv_vgpu_update_request(nvidia_stack_t *, const NvU8 *, NvU32, NvU64 *, NvU64 *, const char *);
NV_STATUS  NV_API_CALL  nv_gpu_bind_event(nvidia_stack_t *);

NV_STATUS NV_API_CALL nv_get_usermap_access_params(nv_state_t*, nv_usermap_access_params_t*);
nv_soc_irq_type_t NV_API_CALL nv_get_current_irq_type(nv_state_t*);

/* Callbacks should occur roughly every 10ms. */
#define NV_SNAPSHOT_TIMER_HZ 100
void NV_API_CALL nv_start_snapshot_timer(void (*snapshot_callback)(void *context));
void NV_API_CALL nv_flush_snapshot_timer(void);
void NV_API_CALL nv_stop_snapshot_timer(void);

static inline const NvU8 *nv_get_cached_uuid(nv_state_t *nv)
{
    return nv->nv_uuid_cache.valid ? nv->nv_uuid_cache.uuid : NULL;
}













#endif /* NVRM */

static inline int nv_count_bits(NvU64 word)
{
    NvU64 bits;

    bits = (word & 0x5555555555555555ULL) + ((word >>  1) & 0x5555555555555555ULL);
    bits = (bits & 0x3333333333333333ULL) + ((bits >>  2) & 0x3333333333333333ULL);
    bits = (bits & 0x0f0f0f0f0f0f0f0fULL) + ((bits >>  4) & 0x0f0f0f0f0f0f0f0fULL);
    bits = (bits & 0x00ff00ff00ff00ffULL) + ((bits >>  8) & 0x00ff00ff00ff00ffULL);
    bits = (bits & 0x0000ffff0000ffffULL) + ((bits >> 16) & 0x0000ffff0000ffffULL);
    bits = (bits & 0x00000000ffffffffULL) + ((bits >> 32) & 0x00000000ffffffffULL);

    return (int)(bits);
}

#endif
