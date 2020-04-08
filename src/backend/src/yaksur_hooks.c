/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <stdlib.h>
#include "yaksa.h"
#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri.h"

yaksuri_global_s yaksuri_global;

int yaksur_init_hook(void)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_seq_init_hook();
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksuri_cuda_init_hook(&yaksuri_global.cuda.host.malloc, &yaksuri_global.cuda.host.free,
                                &yaksuri_global.cuda.device.malloc,
                                &yaksuri_global.cuda.device.free);
    YAKSU_ERR_CHECK(rc, fn_fail);

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

    if (yaksuri_global.cuda.host.slab) {
        yaksuri_global.cuda.host.free(yaksuri_global.cuda.host.slab);
    }
    if (yaksuri_global.cuda.device.slab) {
        yaksuri_global.cuda.device.free(yaksuri_global.cuda.device.slab);
    }

    rc = yaksuri_cuda_finalize_hook();
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_type_create_hook(yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_seq_type_create_hook(type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksuri_cuda_type_create_hook(type);
    YAKSU_ERR_CHECK(rc, fn_fail);

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

    rc = yaksuri_cuda_type_free_hook(type);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_request_create_hook(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_cuda_event_create(&request->backend_priv.event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_request_free_hook(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_cuda_event_destroy(request->backend_priv.event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
