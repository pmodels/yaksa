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

int yaksuri_cuda_request_test(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cuda_request_wait(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
