/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include <assert.h>

int yaksa_request_test(yaksa_request_t request, int *completed)
{
    int rc = YAKSA_SUCCESS;

    assert(yaksi_global.is_initialized);

    assert(request == YAKSA_REQUEST__NULL);
    *completed = 1;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksa_request_wait(yaksa_request_t request)
{
    int rc = YAKSA_SUCCESS;

    assert(yaksi_global.is_initialized);

    assert(request == YAKSA_REQUEST__NULL);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
