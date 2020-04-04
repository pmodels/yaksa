/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksa.h"
#include "yaksi.h"
#include "yaksu.h"

int yaksur_ipack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                 yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;
    yaksur_memory_type_e inbuf_memtype, outbuf_memtype;

    rc = yaksuri_cuda_get_memory_type((const char *) inbuf + type->true_lb, &inbuf_memtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksuri_cuda_get_memory_type(outbuf, &outbuf_memtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (inbuf_memtype != YAKSUR_MEMORY_TYPE__DEVICE && outbuf_memtype != YAKSUR_MEMORY_TYPE__DEVICE) {
        if (type->backend_priv.seq.pack) {
            rc = type->backend_priv.seq.pack(inbuf, outbuf, count, type);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else {
            rc = YAKSA_ERR__NOT_SUPPORTED;
            goto fn_exit;
        }
    } else if (inbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE &&
               outbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE) {
        if (type->backend_priv.cuda.pack) {
            rc = type->backend_priv.cuda.pack(inbuf, outbuf, count, type, request);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else {
            rc = YAKSA_ERR__NOT_SUPPORTED;
            goto fn_exit;
        }
    } else if (inbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE &&
               outbuf_memtype != YAKSUR_MEMORY_TYPE__DEVICE) {
        /* FIXME: pack from D2H is not supported yet */
        rc = YAKSA_ERR__INTERNAL;
        goto fn_exit;
    } else {
        /* FIXME: pack from H2D is not supported yet */
        rc = YAKSA_ERR__INTERNAL;
        goto fn_exit;
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_iunpack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                   yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;
    yaksur_memory_type_e inbuf_memtype, outbuf_memtype;

    rc = yaksuri_cuda_get_memory_type(inbuf, &inbuf_memtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksuri_cuda_get_memory_type((char *) outbuf + type->true_lb, &outbuf_memtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (inbuf_memtype != YAKSUR_MEMORY_TYPE__DEVICE && outbuf_memtype != YAKSUR_MEMORY_TYPE__DEVICE) {
        if (type->backend_priv.seq.unpack) {
            rc = type->backend_priv.seq.unpack(inbuf, outbuf, count, type);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else {
            rc = YAKSA_ERR__NOT_SUPPORTED;
            goto fn_exit;
        }
    } else if (inbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE &&
               outbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE) {
        if (type->backend_priv.cuda.unpack) {
            rc = type->backend_priv.cuda.unpack(inbuf, outbuf, count, type, request);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else {
            rc = YAKSA_ERR__NOT_SUPPORTED;
            goto fn_exit;
        }
    } else if (inbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE &&
               outbuf_memtype != YAKSUR_MEMORY_TYPE__DEVICE) {
        /* FIXME: unpack from D2H is not supported yet */
        rc = YAKSA_ERR__INTERNAL;
        goto fn_exit;
    } else {
        /* FIXME: unpack from H2D is not supported yet */
        rc = YAKSA_ERR__INTERNAL;
        goto fn_exit;
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_request_test(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_cuda_request_test(request);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_request_wait(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

    while (yaksu_atomic_load(&request->cc)) {
        rc = yaksuri_cuda_request_test(request);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
