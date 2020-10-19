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

int yaksuri_cudai_event_destroy(void *event)
{
    int rc = YAKSA_SUCCESS;

    if (event) {
        cudaError_t cerr = cudaEventDestroy((cudaEvent_t) event);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cudai_event_query(void *event, int *completed)
{
    int rc = YAKSA_SUCCESS;

    if (event) {
        cudaError_t cerr = cudaEventQuery((cudaEvent_t) event);
        if (cerr == cudaSuccess) {
            *completed = 1;
        } else if (cerr == cudaErrorNotReady) {
            *completed = 0;
        } else {
            YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
        }
    } else {
        *completed = 1;
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cudai_event_synchronize(void *event)
{
    int rc = YAKSA_SUCCESS;

    if (event) {
        cudaError_t cerr = cudaEventSynchronize((cudaEvent_t) event);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cudai_event_add_dependency(void *event, int device)
{
    int rc = YAKSA_SUCCESS;
    cudaError_t cerr;

    int cur_device;
    cerr = cudaGetDevice(&cur_device);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    if (device != cur_device) {
        cerr = cudaSetDevice(device);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
    }

    cerr = cudaStreamWaitEvent(yaksuri_cudai_global.stream[device], (cudaEvent_t) event, 0);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    if (device != cur_device) {
        cerr = cudaSetDevice(cur_device);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
