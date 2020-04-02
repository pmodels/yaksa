/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include <string.h>
#include <assert.h>

int yaksi_iunpack(const void *inbuf, uintptr_t insize, void *outbuf, uintptr_t outcount,
                  yaksi_type_s * type, uintptr_t outoffset, yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;
    uintptr_t total_bytes = outcount * type->size - outoffset;

    assert(insize <= total_bytes);

    if (type->kind == YAKSI_TYPE_KIND__BUILTIN && insize < type->size)
        goto fn_exit;

    /* We follow these steps:
     *
     *   1. Skip the first few elements till we reach an element that
     *   can contribute some data to our unpacking.
     *
     *   2. Partial unpack the next element.  Once unpacked, if either
     *   the unpack buffer is full or there's data left over in this
     *   element, return.
     *
     *   3. Perform a full unpack of the next few elements.
     *
     *   4. Partial unpack the next element.  Once unpacked, if either
     *   the unpack buffer is full or there's data left over in this
     *   element, return.
     *
     * In the common case, we expect to execute only step 3.
     */

    const char *sbuf = (const char *) inbuf;
    char *dbuf = (char *) outbuf;
    uintptr_t remoffset = outoffset;
    uintptr_t rem_unpack_bytes = YAKSU_MIN(insize, outcount * type->size - outoffset);

    /* step 1: skip the first few elements */
    if (remoffset) {
        uintptr_t skipelems = remoffset / type->size;

        remoffset %= type->size;
        dbuf += skipelems * type->extent;
    }


    /* step 2: partial unpack the next element */
    if (remoffset) {
        assert(type->size > remoffset);

        uintptr_t tmp_unpack_bytes = YAKSU_MIN(rem_unpack_bytes, type->size - remoffset);

        rc = yaksi_iunpack_element(sbuf, tmp_unpack_bytes, dbuf, type, remoffset, request);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rem_unpack_bytes -= tmp_unpack_bytes;

        if (rem_unpack_bytes == 0) {
            /* if we are out of unpack buffer, return */
            goto fn_exit;
        }

        remoffset = 0;
        sbuf += tmp_unpack_bytes;
        dbuf += type->extent;
    }


    /* step 3: perform a full unpack of the next few elements */
    uintptr_t numelems = rem_unpack_bytes / type->size;
    if (numelems) {
        rc = yaksi_iunpack_backend(sbuf, dbuf, numelems, type, request);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rem_unpack_bytes -= numelems * type->size;

        sbuf += numelems * type->size;
        dbuf += numelems * type->extent;
    }


    /* step 4: partial unpack the next element */
    if (rem_unpack_bytes) {
        rc = yaksi_iunpack_element(sbuf, rem_unpack_bytes, dbuf, type, remoffset, request);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
