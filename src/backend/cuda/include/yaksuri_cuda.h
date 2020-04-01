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
#include <cuda.h>
#include <cuda_runtime_api.h>

struct yaksuri_cudai_md_s;
typedef struct {
    struct yaksuri_cudai_md_s *md;
    pthread_mutex_t mdmutex;
    uintptr_t num_elements;
} yaksuri_cuda_type_s;

typedef struct {
    cudaEvent_t event;
} yaksuri_cuda_request_s;

struct yaksi_type_s;
struct yaksi_request_s;

int yaksuri_cuda_init_hook(void);
int yaksuri_cuda_finalize_hook(void);
int yaksuri_cuda_type_create_hook(struct yaksi_type_s *type);
int yaksuri_cuda_type_free_hook(struct yaksi_type_s *type);
int yaksuri_cuda_request_create_hook(struct yaksi_request_s *request);
int yaksuri_cuda_request_free_hook(struct yaksi_request_s *request);
int yaksuri_cuda_request_test(struct yaksi_request_s *request);
int yaksuri_cuda_request_wait(struct yaksi_request_s *request);

int yaksuri_cuda_is_gpu_memory(const void *buf, int *flag);

#endif /* YAKSURI_CUDA_H_INCLUDED */
