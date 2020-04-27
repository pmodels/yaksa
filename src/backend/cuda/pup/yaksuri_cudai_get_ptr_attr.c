/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri_cudai.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

int yaksuri_cudai_get_ptr_attr(const void *buf, yaksur_ptr_attr_s * ptrattr)
{
    int rc = YAKSA_SUCCESS;

    struct cudaPointerAttributes attr;
    cudaError_t cerr = cudaPointerGetAttributes(&attr, buf);
    if (cerr == cudaSuccess) {
        if (attr.type == cudaMemoryTypeUnregistered) {
            ptrattr->type = YAKSUR_PTR_TYPE__UNREGISTERED_HOST;
            ptrattr->device = -1;
        } else if (attr.type == cudaMemoryTypeHost) {
            ptrattr->type = YAKSUR_PTR_TYPE__REGISTERED_HOST;
            ptrattr->device = -1;
        } else {
            ptrattr->type = YAKSUR_PTR_TYPE__GPU;
            ptrattr->device = attr.device;
        }
    } else if (cerr == cudaErrorInvalidValue) {
        ptrattr->type = YAKSUR_PTR_TYPE__UNREGISTERED_HOST;
        ptrattr->device = -1;
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
