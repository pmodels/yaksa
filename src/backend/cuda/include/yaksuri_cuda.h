/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_CUDA_H_INCLUDED
#define YAKSURI_CUDA_H_INCLUDED

#include "yaksi.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

int yaksuri_cuda_init_hook(void);
int yaksuri_cuda_finalize_hook(void);
int yaksuri_cuda_type_create_hook(yaksi_type_s * type);
int yaksuri_cuda_type_free_hook(yaksi_type_s * type);

static int yaksuri_cuda_is_gpu_memory(const void *buf, int *flag)
{
    int rc = YAKSA_SUCCESS;

    struct cudaPointerAttributes attr;
    cudaError_t cerr = cudaPointerGetAttributes(&attr, buf);
    if (cerr == cudaSuccess) {
        if (attr.type == cudaMemoryTypeUnregistered || attr.type == cudaMemoryTypeHost) {
            *flag = 0;
        } else {
            *flag = 1;
        }
    } else if (cerr == cudaErrorInvalidValue) {
        *flag = 0;
    } else if (cerr != cudaSuccess) {
        fprintf(stderr, "CUDA Error (%s:%s,%d): %s\n", __func__, __FILE__, __LINE__,
                cudaGetErrorString(cerr));
        rc = YAKSA_ERR__INTERNAL;
        goto fn_fail;
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

#endif /* YAKSURI_CUDA_H_INCLUDED */
