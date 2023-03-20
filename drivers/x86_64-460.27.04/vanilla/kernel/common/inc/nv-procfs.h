/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */
#ifndef _NV_PROCFS_H
#define _NV_PROCFS_H

#include "conftest.h"

#ifdef CONFIG_PROC_FS
#include <linux/proc_fs.h>
#include <linux/seq_file.h>

/*
 * Allow procfs to create file to exercise error forwarding.
 * This is supported by CRAY platforms.
 */
#if defined(CONFIG_CRAY_XT)
#define EXERCISE_ERROR_FORWARDING NV_TRUE
#else
#define EXERCISE_ERROR_FORWARDING NV_FALSE
#endif

#define IS_EXERCISE_ERROR_FORWARDING_ENABLED() (EXERCISE_ERROR_FORWARDING)

#if defined(NV_PROC_OPS_PRESENT)
typedef struct proc_ops nv_proc_ops_t;

#define NV_PROC_OPS_SET_OWNER()

#define NV_PROC_OPS_OPEN    proc_open
#define NV_PROC_OPS_READ    proc_read
#define NV_PROC_OPS_WRITE   proc_write
#define NV_PROC_OPS_LSEEK   proc_lseek
#define NV_PROC_OPS_RELEASE proc_release
#else
typedef struct file_operations nv_proc_ops_t;

#define NV_PROC_OPS_SET_OWNER() .owner = THIS_MODULE,

#define NV_PROC_OPS_OPEN    open
#define NV_PROC_OPS_READ    read
#define NV_PROC_OPS_WRITE   write
#define NV_PROC_OPS_LSEEK   llseek
#define NV_PROC_OPS_RELEASE release
#endif

#define NV_CREATE_PROC_FILE(filename,parent,__name,__data)               \
   ({                                                                    \
        struct proc_dir_entry *__entry;                                  \
        int mode = (S_IFREG | S_IRUGO);                                  \
        const nv_proc_ops_t *fops = &nv_procfs_##__name##_fops;          \
        if (fops->NV_PROC_OPS_WRITE != 0)                                \
            mode |= S_IWUSR;                                             \
        __entry = proc_create_data(filename, mode, parent, fops, __data);\
        __entry;                                                         \
    })

/*
 * proc_mkdir_mode exists in Linux 2.6.9, but isn't exported until Linux 3.0.
 * Use the older interface instead unless the newer interface is necessary.
 */
#if defined(NV_PROC_REMOVE_PRESENT)
# define NV_PROC_MKDIR_MODE(name, mode, parent)                \
    proc_mkdir_mode(name, mode, parent)
#else
# define NV_PROC_MKDIR_MODE(name, mode, parent)                \
   ({                                                          \
        struct proc_dir_entry *__entry;                        \
        __entry = create_proc_entry(name, mode, parent);       \
        __entry;                                               \
    })
#endif

#define NV_CREATE_PROC_DIR(name,parent)                        \
   ({                                                          \
        struct proc_dir_entry *__entry;                        \
        int mode = (S_IFDIR | S_IRUGO | S_IXUGO);              \
        __entry = NV_PROC_MKDIR_MODE(name, mode, parent);      \
        __entry;                                               \
    })

#if defined(NV_PDE_DATA_PRESENT)
# define NV_PDE_DATA(inode) PDE_DATA(inode)
#else
# define NV_PDE_DATA(inode) PDE(inode)->data
#endif

#if defined(NV_PROC_REMOVE_PRESENT)
# define NV_REMOVE_PROC_ENTRY(entry)                           \
    proc_remove(entry);
#else
# define NV_REMOVE_PROC_ENTRY(entry)                           \
    remove_proc_entry(entry->name, entry->parent);
#endif

void nv_procfs_unregister_all(struct proc_dir_entry *entry,
                              struct proc_dir_entry *delimiter);

#define NV_DEFINE_SINGLE_PROCFS_FILE(name, open_callback, close_callback)     \
    static int nv_procfs_open_##name(                                         \
        struct inode *inode,                                                  \
        struct file *filep                                                    \
    )                                                                         \
    {                                                                         \
        int ret;                                                              \
        ret = single_open(filep, nv_procfs_read_##name,                       \
                          NV_PDE_DATA(inode));                                \
        if (ret < 0)                                                          \
        {                                                                     \
            return ret;                                                       \
        }                                                                     \
        ret = open_callback();                                                \
        if (ret < 0)                                                          \
        {                                                                     \
            single_release(inode, filep);                                     \
        }                                                                     \
        return ret;                                                           \
    }                                                                         \
                                                                              \
    static int nv_procfs_release_##name(                                      \
        struct inode *inode,                                                  \
        struct file *filep                                                    \
    )                                                                         \
    {                                                                         \
        close_callback();                                                     \
        return single_release(inode, filep);                                  \
    }                                                                         \
                                                                              \
    static const nv_proc_ops_t nv_procfs_##name##_fops = {                    \
        NV_PROC_OPS_SET_OWNER()                                               \
        .NV_PROC_OPS_OPEN    = nv_procfs_open_##name,                         \
        .NV_PROC_OPS_READ    = seq_read,                                      \
        .NV_PROC_OPS_LSEEK   = seq_lseek,                                     \
        .NV_PROC_OPS_RELEASE = nv_procfs_release_##name,                      \
    };

#endif  /* CONFIG_PROC_FS */

#endif /* _NV_PROCFS_H */
