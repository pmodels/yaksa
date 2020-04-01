/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSUR_H_INCLUDED
#define YAKSUR_H_INCLUDED

/* This is a API header exposed by the backend glue layer.  It should
 * not include any internal headers except: (1) yaksa_config.h, in
 * order to get the configure checks; and (2) API headers for the
 * devices (e.g., yaksuri_seq.h) */
#include <stdint.h>
#include "yaksa_config.h"
#include "yaksuri_seq.h"

#ifdef HAVE_CUDA
#include "yaksuri_cuda.h"
#endif /* HAVE_CUDA */

struct yaksi_type_s;
struct yaksi_request_s;

typedef int (*yaksur_pup_fn) (const void *, void *, uintptr_t, struct yaksi_type_s *,
                              struct yaksi_request_s *);

typedef struct yaksur_type_s {
    struct {
        yaksur_pup_fn pack;
        yaksur_pup_fn unpack;
    } seq;

#ifdef HAVE_CUDA
    struct {
        yaksur_pup_fn pack;
        yaksur_pup_fn unpack;
    } cuda;
#endif                          /* HAVE_CUDA */

    /* give some private space to each backend to store content */
    yaksuri_seq_type_s seq_priv;

#ifdef HAVE_CUDA
    yaksuri_cuda_type_s cuda_priv;
#endif                          /* HAVE_CUDA */
} yaksur_type_s;

typedef struct {
    /* give some private space to each backend to store content */
    yaksuri_seq_request_s seq_priv;

#ifdef HAVE_CUDA
    yaksuri_cuda_request_s cuda_priv;
#endif                          /* HAVE_CUDA */
} yaksur_request_s;

int yaksur_init_hook(void);
int yaksur_finalize_hook(void);
int yaksur_type_create_hook(struct yaksi_type_s *type);
int yaksur_type_free_hook(struct yaksi_type_s *type);
int yaksur_request_create_hook(struct yaksi_request_s *request);
int yaksur_request_free_hook(struct yaksi_request_s *request);

int yaksur_ipack(const void *inbuf, void *outbuf, uintptr_t count, struct yaksi_type_s *type,
                 struct yaksi_request_s *request);
int yaksur_iunpack(const void *inbuf, void *outbuf, uintptr_t count, struct yaksi_type_s *type,
                   struct yaksi_request_s *request);
int yaksur_request_test(struct yaksi_request_s *request);
int yaksur_request_wait(struct yaksi_request_s *request);

#endif /* YAKSUR_H_INCLUDED */
