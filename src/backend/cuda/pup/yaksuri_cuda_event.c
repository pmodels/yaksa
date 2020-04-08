/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri_cudai.h"

int yaksuri_cuda_event_create(yaksuri_cuda_event_t * event)
{
    int rc = YAKSA_SUCCESS;

    cudaError_t cerr = cudaEventCreate(event);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cuda_event_destroy(yaksuri_cuda_event_t event)
{
    int rc = YAKSA_SUCCESS;

    cudaError_t cerr = cudaEventDestroy(event);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cuda_event_query(yaksuri_cuda_event_t event, int *completed)
{
    int rc = YAKSA_SUCCESS;

    cudaError_t cerr = cudaEventQuery(event);
    if (cerr == cudaSuccess) {
        *completed = 1;
    } else if (cerr == cudaErrorNotReady) {
        *completed = 0;
    } else {
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cuda_event_synchronize(yaksuri_cuda_event_t event)
{
    int rc = YAKSA_SUCCESS;

    cudaError_t cerr = cudaEventSynchronize(event);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
