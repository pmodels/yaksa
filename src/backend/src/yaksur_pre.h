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

typedef enum {
    YAKSUR_MEMORY_TYPE__UNREGISTERED_HOST,
    YAKSUR_MEMORY_TYPE__REGISTERED_HOST,
    YAKSUR_MEMORY_TYPE__DEVICE,
} yaksur_memory_type_e;

struct yaksi_type_s;
typedef int (*yaksur_seq_pup_fn) (const void *inbuf, void *outbuf, uintptr_t count,
                                  struct yaksi_type_s * type);
typedef int (*yaksur_gpudev_pup_fn) (const void *inbuf, void *outbuf, uintptr_t count,
                                     struct yaksi_type_s * type, void *device_tmpbuf, void *event);

typedef struct yaksur_type_s {
    void *priv;
    yaksuri_seq_type_s seq;
    yaksuri_cuda_type_s cuda;
} yaksur_type_s;

typedef struct {
    void *priv;
} yaksur_request_s;

#endif /* YAKSUR_PRE_H_INCLUDED */
