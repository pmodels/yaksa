/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri_cudai.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

int yaksuri_cuda_request_create_hook(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

    cudaError_t cerr = cudaEventCreate(&request->backend_priv.cuda_priv.event);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cuda_request_free_hook(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

    cudaError_t cerr = cudaEventDestroy(request->backend_priv.cuda_priv.event);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
