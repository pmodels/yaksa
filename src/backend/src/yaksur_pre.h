/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSUR_PRE_H_INCLUDED
#define YAKSUR_PRE_H_INCLUDED

/* This is a API header exposed by the backend glue layer.  It should
 * not include any internal headers except: (1) yaksa_config.h, in
 * order to get the configure checks; and (2) API headers for the
 * devices (e.g., yaksuri_seq.h) */
#include <stdint.h>
#include "yaksuri_seq_pre.h"
#include "yaksuri_cuda_pre.h"

typedef struct {
    enum {
        YAKSUR_PTR_TYPE__UNREGISTERED_HOST,
        YAKSUR_PTR_TYPE__REGISTERED_HOST,
        YAKSUR_PTR_TYPE__GPU,
    } type;
    int device;
} yaksur_ptr_attr_s;

struct yaksi_type_s;
struct yaksi_info_s;

typedef struct yaksur_type_s {
    yaksuri_seq_type_s seq;
    yaksuri_cuda_type_s cuda;
} yaksur_type_s;

typedef struct {
    void *priv;
} yaksur_request_s;

typedef struct {
    yaksuri_seq_info_s seq;
    yaksuri_cuda_info_s cuda;
} yaksur_info_s;

typedef struct yaksur_gpudriver_info_s {
    /* miscellaneous */
    int (*get_num_devices) (int *ndevices);
    int (*check_p2p_comm) (int sdev, int ddev, bool * is_enabled);
    int (*finalize) (void);

    /* pup functions */
    int (*ipack) (const void *inbuf, void *outbuf, uintptr_t count,
                  struct yaksi_type_s * type, struct yaksi_info_s * info, void **event,
                  void *device_tmpbuf, int device);
    int (*iunpack) (const void *inbuf, void *outbuf, uintptr_t count, struct yaksi_type_s * type,
                    struct yaksi_info_s * info, void **event, void *device_tmpbuf, int device);
    int (*pup_is_supported) (struct yaksi_type_s * type, bool * is_supported);

    /* memory management */
    void *(*host_malloc) (uintptr_t size);
    void (*host_free) (void *ptr);
    void *(*gpu_malloc) (uintptr_t size, int device);
    void (*gpu_free) (void *ptr);
    int (*get_ptr_attr) (const void *buf, yaksur_ptr_attr_s * ptrattr);

    /* events */
    int (*event_destroy) (void *event);
    int (*event_query) (void *event, int *completed);
    int (*event_synchronize) (void *event);
    int (*event_add_dependency) (void *event, int device);

    /* types */
    int (*type_create) (struct yaksi_type_s * type);
    int (*type_free) (struct yaksi_type_s * type);

    /* info */
    int (*info_create) (struct yaksi_info_s * info);
    int (*info_free) (struct yaksi_info_s * info);
    int (*info_keyval_append) (struct yaksi_info_s * info, const char *key, const void *val,
                               unsigned int vallen);
} yaksur_gpudriver_info_s;

#endif /* YAKSUR_PRE_H_INCLUDED */
