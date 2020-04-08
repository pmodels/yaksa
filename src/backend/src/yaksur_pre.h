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

typedef int (*yaksur_event_create_fn) (void **);
typedef int (*yaksur_event_destroy_fn) (void *);
typedef int (*yaksur_event_query_fn) (void *, int *);
typedef int (*yaksur_event_synchronize_fn) (void *);
typedef int (*yaksur_finalize_fn) (void);
typedef int (*yaksur_type_create_fn) (struct yaksi_type_s *, yaksur_gpudev_pup_fn *,
                                      yaksur_gpudev_pup_fn *);
typedef int (*yaksur_type_free_fn) (struct yaksi_type_s *);
typedef int (*yaksur_get_memory_type) (const void *, yaksur_memory_type_e *);

typedef struct yaksur_gpudev_info_s {
    yaksu_malloc_fn host_malloc;
    yaksu_free_fn host_free;
    yaksu_malloc_fn device_malloc;
    yaksu_free_fn device_free;
    yaksur_event_create_fn event_create;
    yaksur_event_destroy_fn event_destroy;
    yaksur_event_query_fn event_query;
    yaksur_event_synchronize_fn event_synchronize;
    yaksur_type_create_fn type_create;
    yaksur_type_free_fn type_free;
    yaksur_get_memory_type get_memory_type;
    yaksur_finalize_fn finalize;
} yaksur_gpudev_info_s;

#endif /* YAKSUR_PRE_H_INCLUDED */
