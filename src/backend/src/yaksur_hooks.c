/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <stdlib.h>
#include "yaksa.h"
#include "yaksi.h"
#include "yaksu.h"
#include "yaksur.h"
#include "yaksuri.h"

int yaksur_init_hook(void)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_seq_init_hook();
    YAKSU_ERR_CHECK(rc, fn_fail);

#ifdef HAVE_CUDA
    rc = yaksuri_cuda_init_hook();
    YAKSU_ERR_CHECK(rc, fn_fail);
#endif /* HAVE_CUDA */

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_finalize_hook(void)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_seq_finalize_hook();
    YAKSU_ERR_CHECK(rc, fn_fail);

#ifdef HAVE_CUDA
    rc = yaksuri_cuda_finalize_hook();
    YAKSU_ERR_CHECK(rc, fn_fail);
#endif /* HAVE_CUDA */

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_type_create_hook(yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;

    type->backend = malloc(sizeof(yaksuri_type_s));
    YAKSU_ERR_CHKANDJUMP(!type->backend, rc, YAKSA_ERR__OUT_OF_MEM, fn_fail);

    rc = yaksuri_seq_type_create_hook(type);
    YAKSU_ERR_CHECK(rc, fn_fail);

#ifdef HAVE_CUDA
    rc = yaksuri_cuda_type_create_hook(type);
    YAKSU_ERR_CHECK(rc, fn_fail);
#endif /* HAVE_CUDA */

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_type_free_hook(yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_seq_type_free_hook(type);
    YAKSU_ERR_CHECK(rc, fn_fail);

#ifdef HAVE_CUDA
    rc = yaksuri_cuda_type_free_hook(type);
    YAKSU_ERR_CHECK(rc, fn_fail);
#endif /* HAVE_CUDA */

    free(type->backend);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
