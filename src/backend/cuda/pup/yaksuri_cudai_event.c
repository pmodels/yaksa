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

int yaksuri_cudai_event_create(int device, void **event_)
{
    int rc = YAKSA_SUCCESS;
    cudaError_t cerr;
    yaksuri_cudai_event_s *event;

    event = (yaksuri_cudai_event_s *) malloc(sizeof(yaksuri_cudai_event_s));

    int cur_device;
    cerr = cudaGetDevice(&cur_device);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    cerr = cudaSetDevice(device);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    cerr = cudaEventCreate(&event->event);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    cerr = cudaSetDevice(cur_device);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    event->device = device;

    *event_ = event;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cudai_event_destroy(void *event_)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_event_s *event = (yaksuri_cudai_event_s *) event_;

    cudaError_t cerr = cudaEventDestroy(event->event);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    free(event);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cudai_event_record(void *event_)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_event_s *event = (yaksuri_cudai_event_s *) event_;

    cudaError_t cerr = cudaEventRecord(event->event, yaksuri_cudai_global.stream[event->device]);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cudai_event_query(void *event_, int *completed)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_event_s *event = (yaksuri_cudai_event_s *) event_;

    cudaError_t cerr = cudaEventQuery(event->event);
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

int yaksuri_cudai_event_synchronize(void *event_)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_event_s *event = (yaksuri_cudai_event_s *) event_;

    cudaError_t cerr = cudaEventSynchronize(event->event);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cudai_event_add_dependency(void *event_, int device)
{
    int rc = YAKSA_SUCCESS;
    cudaError_t cerr;
    yaksuri_cudai_event_s *event = (yaksuri_cudai_event_s *) event_;

    int cur_device;
    cerr = cudaGetDevice(&cur_device);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    cerr = cudaSetDevice(device);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    cerr = cudaStreamWaitEvent(yaksuri_cudai_global.stream[device], event->event, 0);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    cerr = cudaSetDevice(cur_device);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
