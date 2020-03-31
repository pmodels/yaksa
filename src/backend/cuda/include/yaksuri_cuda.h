/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_CUDA_H_INCLUDED
#define YAKSURI_CUDA_H_INCLUDED

#include "yaksi.h"

int yaksuri_cuda_init_hook(void);
int yaksuri_cuda_finalize_hook(void);
int yaksuri_cuda_type_create_hook(yaksi_type_s * type);
int yaksuri_cuda_type_free_hook(yaksi_type_s * type);
int yaksuri_cuda_is_gpu_memory(const void *buf, int *flag);

#endif /* YAKSURI_CUDA_H_INCLUDED */
