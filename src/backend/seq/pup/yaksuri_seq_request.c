/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <stdio.h>
#include "yaksi.h"
#include "yaksu.h"

int yaksuri_seq_request_test(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_seq_request_wait(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
