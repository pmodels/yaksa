/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"

int yaksi_request_create(yaksi_request_s ** request)
{
    int rc = YAKSA_SUCCESS;
    int idx;

    rc = yaksu_pool_elem_alloc(yaksi_global.request_pool, (void **) request, &idx);
    YAKSU_ERR_CHECK(rc, fn_fail);

    (*request)->id = (yaksa_request_t) idx;
    yaksu_atomic_store(&(*request)->cc, 0);

    rc = yaksur_request_create_hook(*request);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksi_request_free(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksur_request_free_hook(request);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksu_pool_elem_free(yaksi_global.request_pool, request->id);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksi_request_get(yaksa_request_t request, struct yaksi_request_s **yaksi_request)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksu_pool_elem_get(yaksi_global.request_pool, (int) request, (void **) yaksi_request);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
