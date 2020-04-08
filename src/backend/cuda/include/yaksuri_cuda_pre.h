/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_CUDA_PRE_H_INCLUDED
#define YAKSURI_CUDA_PRE_H_INCLUDED

/* This is a API header for the cuda device and should not include any
 * internal headers, except for yaksa_config.h, in order to get the
 * configure checks. */

#include <stdint.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

struct yaksuri_cudai_md_s;
typedef struct {
    struct yaksuri_cudai_md_s *md;
    pthread_mutex_t mdmutex;
    uintptr_t num_elements;
} yaksuri_cuda_type_s;

typedef cudaEvent_t yaksuri_cuda_event_t;

#endif /* YAKSURI_CUDA_PRE_H_INCLUDED */
