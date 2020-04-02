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
#include "yaksa_config.h"
#include "yaksuri_seq_pre.h"
#include "yaksuri_cuda_pre.h"

typedef enum {
    YAKSUR_MEMORY_TYPE__UNREGISTERED_HOST,
    YAKSUR_MEMORY_TYPE__REGISTERED_HOST,
    YAKSUR_MEMORY_TYPE__DEVICE,
} yaksur_memory_type_e;

struct yaksi_type_s;
struct yaksi_request_s;
typedef int (*yaksur_pup_fn) (const void *, void *, uintptr_t, struct yaksi_type_s *,
                              struct yaksi_request_s *);

typedef struct yaksur_type_s {
    struct {
        yaksur_pup_fn pack;
        yaksur_pup_fn unpack;
    } seq;

    struct {
        yaksur_pup_fn pack;
        yaksur_pup_fn unpack;
    } cuda;

    /* give some private space to each backend to store content */
    yaksuri_seq_type_s seq_priv;
    yaksuri_cuda_type_s cuda_priv;
} yaksur_type_s;

typedef struct {
    /* give some private space to each backend to store content */
    yaksuri_seq_request_s seq_priv;
    yaksuri_cuda_request_s cuda_priv;
} yaksur_request_s;

#endif /* YAKSUR_PRE_H_INCLUDED */
