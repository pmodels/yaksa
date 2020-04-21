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
        YAKSUR_PTR_TYPE__DEVICE,
    } type;
    int device;
} yaksur_ptr_attr_s;

struct yaksi_type_s;

typedef struct yaksur_type_s {
    void *priv;
    yaksuri_seq_type_s seq;
    yaksuri_cuda_type_s cuda;
} yaksur_type_s;

typedef struct {
    void *priv;
} yaksur_request_s;

typedef void *(*yaksur_device_malloc_fn) (uintptr_t, int);
typedef int (*yaksur_get_num_devices_fn) (int *);
typedef int (*yaksur_check_p2p_comm_fn) (int, int, bool *);
typedef int (*yaksur_pup_fn) (const void *, void *, uintptr_t, struct yaksi_type_s *,
                              void *, void *, void **);
typedef int (*yaksur_pup_is_supported_fn) (struct yaksi_type_s *, bool *);
typedef int (*yaksur_event_destroy_fn) (void *);
typedef int (*yaksur_event_query_fn) (void *, int *);
typedef int (*yaksur_event_synchronize_fn) (void *);
typedef int (*yaksur_finalize_fn) (void);
typedef int (*yaksur_type_create_fn) (struct yaksi_type_s *);
typedef int (*yaksur_type_free_fn) (struct yaksi_type_s *);
typedef int (*yaksur_get_ptr_attr) (const void *, yaksur_ptr_attr_s *);

typedef struct yaksur_gpudev_info_s {
    yaksur_get_num_devices_fn get_num_devices;
    yaksur_check_p2p_comm_fn check_p2p_comm;
    yaksur_pup_fn ipack;
    yaksur_pup_fn iunpack;
    yaksur_pup_is_supported_fn pup_is_supported;
    yaksu_malloc_fn host_malloc;
    yaksu_free_fn host_free;
    yaksur_device_malloc_fn device_malloc;
    yaksu_free_fn device_free;
    yaksur_event_destroy_fn event_destroy;
    yaksur_event_query_fn event_query;
    yaksur_event_synchronize_fn event_synchronize;
    yaksur_type_create_fn type_create;
    yaksur_type_free_fn type_free;
    yaksur_get_ptr_attr get_ptr_attr;
    yaksur_finalize_fn finalize;
} yaksur_gpudev_info_s;

#endif /* YAKSUR_PRE_H_INCLUDED */
