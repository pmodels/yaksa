/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_CUDA_POST_H_INCLUDED
#define YAKSURI_CUDA_POST_H_INCLUDED

int yaksuri_cuda_init_hook(void);
int yaksuri_cuda_finalize_hook(void);
int yaksuri_cuda_type_create_hook(yaksi_type_s * type);
int yaksuri_cuda_type_free_hook(yaksi_type_s * type);
int yaksuri_cuda_request_create_hook(yaksi_request_s * request);
int yaksuri_cuda_request_free_hook(yaksi_request_s * request);
int yaksuri_cuda_request_test(yaksi_request_s * request);
int yaksuri_cuda_request_wait(yaksi_request_s * request);
int yaksuri_cuda_get_memory_type(const void *buf, yaksur_memory_type_e * memtype);

#endif /* YAKSURI_CUDA_H_INCLUDED */
