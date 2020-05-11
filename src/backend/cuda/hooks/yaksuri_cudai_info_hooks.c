/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksuri_cudai.h"

int yaksuri_cudai_info_create_hook(yaksi_info_s * info)
{
    int rc = YAKSA_SUCCESS;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cudai_info_free_hook(yaksi_info_s * info)
{
    int rc = YAKSA_SUCCESS;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cudai_info_keyval_append(yaksi_info_s * info, const char *key, const void *val,
                                     unsigned int vallen)
{
    int rc = YAKSA_SUCCESS;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
