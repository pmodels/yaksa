/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include <assert.h>

int yaksa_iacc(const void *inbuf, uintptr_t incount, yaksa_type_t intype, uintptr_t inoffset,
               void *outbuf, uintptr_t outcount, yaksa_type_t outtype, uintptr_t outoffset,
               uintptr_t max_acc_bytes, yaksa_info_t info, yaksa_op_t op, yaksa_request_t * request)
{
    int rc = YAKSA_SUCCESS;

    assert(yaksu_atomic_load(&yaksi_is_initialized));

    if (incount == 0) {
        *request = YAKSA_REQUEST__NULL;
        goto fn_exit;
    }

    yaksi_type_s *yaksi_intype, *yaksi_outtype;
    rc = yaksi_type_get(intype, &yaksi_intype);
    YAKSU_ERR_CHECK(rc, fn_fail);
    rc = yaksi_type_get(outtype, &yaksi_outtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    assert(yaksi_intype->size * incount >= inoffset);
    assert(yaksi_outtype->size * outcount >= outoffset);
    assert(yaksi_intype->size * incount - inoffset <= yaksi_outtype->size * outcount - outoffset);

    if (yaksi_intype->size * incount == inoffset) {
        *request = YAKSA_REQUEST__NULL;
        goto fn_exit;
    }

    yaksi_request_s *yaksi_request;
    rc = yaksi_request_create(&yaksi_request);
    YAKSU_ERR_CHECK(rc, fn_fail);

    yaksi_info_s *yaksi_info;
    yaksi_info = (yaksi_info_s *) info;

    if (yaksi_outtype->is_contig) {
        uintptr_t actual_pack_bytes;
        void *real_outbuf = (void *) ((char *) outbuf + yaksi_outtype->true_lb + outoffset);

        rc = yaksi_ipack(inbuf, incount, yaksi_intype, inoffset, real_outbuf,
                         yaksi_intype->size * incount - inoffset,
                         &actual_pack_bytes, yaksi_info, op, yaksi_request);
        YAKSU_ERR_CHECK(rc, fn_fail);
        assert(actual_pack_bytes == yaksi_intype->size * incount - inoffset);
    } else if (yaksi_intype->is_contig) {
        uintptr_t actual_unpack_bytes;
        const void *real_inbuf =
            (const void *) ((const char *) inbuf + yaksi_intype->true_lb + inoffset);

        rc = yaksi_iunpack(real_inbuf, yaksi_intype->size * incount - inoffset, outbuf, outcount,
                           yaksi_outtype, outoffset, &actual_unpack_bytes, yaksi_info, op,
                           yaksi_request);
        YAKSU_ERR_CHECK(rc, fn_fail);
        assert(actual_unpack_bytes == yaksi_intype->size * incount - inoffset);
    } else {
        assert(0);
    }

    if (yaksu_atomic_load(&yaksi_request->cc)) {
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
