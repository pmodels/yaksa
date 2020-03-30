/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_H_INCLUDED
#define YAKSURI_H_INCLUDED

#include <stdint.h>
#include "yaksi.h"
#include "yaksuri_seq.h"

#ifdef HAVE_CUDA
#include "yaksuri_cuda.h"
#endif /* HAVE_CUDA */


typedef int (*pup_fn) (const void *, void *, uintptr_t, yaksi_type_s *, yaksi_request_s **);

struct pupfns {
    pup_fn pack;
    pup_fn unpack;
};

typedef struct {
    struct pupfns seq;

#ifdef HAVE_CUDA
    struct pupfns cuda;
#endif                          /* HAVE_CUDA */

    /* give some space to each backend to store content */
    void *seq_priv;

#ifdef HAVE_CUDA
    void *cuda_priv;
#endif                          /* HAVE_CUDA */
} yaksuri_type_s;

#endif /* YAKSURI_H_INCLUDED */
