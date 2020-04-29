/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri_seqi.h"
#include <stdlib.h>

int yaksuri_seq_init_hook(void)
{
    return YAKSA_SUCCESS;
}

int yaksuri_seq_finalize_hook(void)
{
    return YAKSA_SUCCESS;
}

int yaksuri_seq_type_create_hook(yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;

    type->backend.seq.priv = malloc(sizeof(yaksuri_seqi_type_s));

    rc = yaksuri_seqi_populate_pupfns(type);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_seq_type_free_hook(yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;

    free(type->backend.seq.priv);

    return rc;
}
