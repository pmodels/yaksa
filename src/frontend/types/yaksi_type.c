/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include <assert.h>

int yaksi_type_handle_alloc(yaksi_type_s * type, uint32_t * handle)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksu_handle_pool_elem_alloc(yaksi_global.type_handle_pool, handle, type);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksi_type_handle_dealloc(uint32_t handle, yaksi_type_s ** type)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksu_handle_pool_elem_get(yaksi_global.type_handle_pool, handle, (const void **) type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksu_handle_pool_elem_free(yaksi_global.type_handle_pool, handle);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksi_type_get(yaksa_type_t type, struct yaksi_type_s **yaksi_type)
{
    int rc = YAKSA_SUCCESS;
    uint32_t id = (uint32_t) ((type << 32) >> 32);

    rc = yaksu_handle_pool_elem_get(yaksi_global.type_handle_pool, id, (const void **) yaksi_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
