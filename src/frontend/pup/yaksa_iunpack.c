/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int yaksa_iunpack(const void *inbuf, uintptr_t insize, void *outbuf, uintptr_t outcount,
                  yaksa_type_t type, uintptr_t outoffset, uintptr_t * actual_unpack_bytes,
                  yaksa_request_t * request)
{
    int rc = YAKSA_SUCCESS;

    assert(yaksi_global.is_initialized);

    if (outcount == 0) {
        *request = YAKSA_REQUEST__NULL;
        goto fn_exit;
    }

    yaksi_type_s *yaksi_type;
    rc = yaksi_type_get(type, &yaksi_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (yaksi_type->size == 0) {
        *request = YAKSA_REQUEST__NULL;
        goto fn_exit;
    }

    yaksi_request_s *yaksi_request = NULL;
    rc = yaksi_request_create(&yaksi_request);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksi_iunpack(inbuf, insize, outbuf, outcount, yaksi_type, outoffset, actual_unpack_bytes,
                       yaksi_request);
    YAKSU_ERR_CHECK(rc, fn_fail);

    int cc = yaksu_atomic_load(&yaksi_request->cc);
    if (cc) {
        *request = yaksi_request->id;
    } else {
        rc = yaksi_request_free(yaksi_request);
        YAKSU_ERR_CHECK(rc, fn_fail);

        *request = YAKSA_REQUEST__NULL;
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
