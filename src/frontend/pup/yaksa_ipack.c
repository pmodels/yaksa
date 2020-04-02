/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include <assert.h>

int yaksa_ipack(const void *inbuf, uintptr_t incount, yaksa_type_t type, uintptr_t inoffset,
                void *outbuf, uintptr_t max_pack_bytes, uintptr_t * actual_pack_bytes,
                yaksa_request_t * request)
{
    int rc = YAKSA_SUCCESS;

    assert(yaksi_global.is_initialized);

    yaksi_type_s *yaksi_type;
    rc = yaksi_type_get(type, &yaksi_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    yaksi_request_s *yaksi_request;
    rc = yaksi_request_create(&yaksi_request);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksi_ipack(inbuf, incount, yaksi_type, inoffset, outbuf, max_pack_bytes,
                     actual_pack_bytes, yaksi_request);
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
