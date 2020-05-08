/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"

int yaksi_type_alloc(struct yaksi_type_s **type)
{
    int rc = YAKSA_SUCCESS;
    int idx;

    rc = yaksu_pool_elem_alloc(yaksi_global.type_pool, (void **) type, &idx);
    YAKSU_ERR_CHECK(rc, fn_fail);

    (*type)->id = idx;
    yaksu_atomic_store(&(*type)->refcount, 1);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksi_type_dealloc(struct yaksi_type_s *type)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksu_pool_elem_free(yaksi_global.type_pool, type->id);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksi_type_get(yaksa_type_t type, struct yaksi_type_s **yaksi_type)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksu_pool_elem_get(yaksi_global.type_pool, (int) type, (void **) yaksi_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
