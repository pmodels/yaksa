/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_CUDA_H_INCLUDED
#define YAKSURI_CUDA_H_INCLUDED

/* This is a API header for the cuda device and should not include any
 * internal headers, except for yaksa_config.h, in order to get the
 * configure checks. */

#include <stdint.h>
#include <pthread.h>

struct yaksuri_cudai_md_s;
typedef struct {
    struct yaksuri_cudai_md_s *md;
    pthread_mutex_t mdmutex;
    uintptr_t num_elements;
} yaksuri_cuda_type_s;

struct yaksi_type_s;

int yaksuri_cuda_init_hook(void);
int yaksuri_cuda_finalize_hook(void);
int yaksuri_cuda_type_create_hook(struct yaksi_type_s *type);
int yaksuri_cuda_type_free_hook(struct yaksi_type_s *type);
int yaksuri_cuda_is_gpu_memory(const void *buf, int *flag);

#endif /* YAKSURI_CUDA_H_INCLUDED */
