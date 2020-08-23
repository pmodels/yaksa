/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include <assert.h>

int yaksi_type_handle_alloc(yaksi_type_s * type, yaksu_handle_t * handle)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksu_handle_pool_elem_alloc(yaksi_global.type_handle_pool, handle, type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    assert(*handle < ((yaksa_type_t) 1 << YAKSI_TYPE_OBJECT_ID_BITS));

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksi_type_handle_dealloc(yaksu_handle_t handle, yaksi_type_s ** type)
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
    yaksu_handle_t id = YAKSI_TYPE_GET_OBJECT_ID(type);

    if (id < YAKSI_TYPE__LAST) {
        assert(yaksi_global.yaksi_builtin_types[id]);
        *yaksi_type = yaksi_global.yaksi_builtin_types[id];
    } else {
        rc = yaksu_handle_pool_elem_get(yaksi_global.type_handle_pool, id,
                                        (const void **) yaksi_type);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
