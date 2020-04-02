/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

int yaksuri_cuda_get_memory_type(const void *buf, yaksur_memory_type_e * memtype)
{
    int rc = YAKSA_SUCCESS;

    struct cudaPointerAttributes attr;
    cudaError_t cerr = cudaPointerGetAttributes(&attr, buf);
    if (cerr == cudaSuccess) {
        if (attr.type == cudaMemoryTypeUnregistered) {
            *memtype = YAKSUR_MEMORY_TYPE__UNREGISTERED_HOST;
        } else if (attr.type == cudaMemoryTypeHost) {
            *memtype = YAKSUR_MEMORY_TYPE__REGISTERED_HOST;
        } else {
            *memtype = YAKSUR_MEMORY_TYPE__DEVICE;
        }
    } else if (cerr == cudaErrorInvalidValue) {
        *memtype = YAKSUR_MEMORY_TYPE__UNREGISTERED_HOST;
    } else {
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
