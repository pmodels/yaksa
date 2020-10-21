/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <assert.h>
#include "yaksa.h"
#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri.h"

static int get_ptr_attr(const void *buf, yaksur_ptr_attr_s * ptrattr, yaksuri_gpudriver_id_e * id)
{
    int rc = YAKSA_SUCCESS;

    /* Each GPU driver can claim "ownership" of the input buffer */
    for (*id = YAKSURI_GPUDRIVER_ID__UNSET; *id < YAKSURI_GPUDRIVER_ID__LAST; (*id)++) {
        if (*id == YAKSURI_GPUDRIVER_ID__UNSET)
            continue;

        if (yaksuri_global.gpudriver[*id].hooks) {
            rc = yaksuri_global.gpudriver[*id].hooks->get_ptr_attr(buf, ptrattr);
            YAKSU_ERR_CHECK(rc, fn_fail);

            if (ptrattr->type == YAKSUR_PTR_TYPE__GPU ||
                ptrattr->type == YAKSUR_PTR_TYPE__REGISTERED_HOST)
                break;
        }
    }

    if (*id == YAKSURI_GPUDRIVER_ID__LAST) {
        *id = YAKSURI_GPUDRIVER_ID__UNSET;
        ptrattr->type = YAKSUR_PTR_TYPE__UNREGISTERED_HOST;
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int ipup(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                yaksi_info_s * info, yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e inbuf_gpudriver, outbuf_gpudriver;
    yaksuri_request_s *reqpriv = (yaksuri_request_s *) request->backend.priv;

    if (reqpriv->optype == YAKSURI_OPTYPE__PACK) {
        rc = get_ptr_attr((const char *) inbuf + type->true_lb, &request->backend.inattr,
                          &inbuf_gpudriver);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = get_ptr_attr(outbuf, &request->backend.outattr, &outbuf_gpudriver);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        rc = get_ptr_attr(inbuf, &request->backend.inattr, &inbuf_gpudriver);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = get_ptr_attr((char *) outbuf + type->true_lb, &request->backend.outattr,
                          &outbuf_gpudriver);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    /* if this can be handled by the CPU, wrap it up */
    if (request->backend.inattr.type != YAKSUR_PTR_TYPE__GPU &&
        request->backend.outattr.type != YAKSUR_PTR_TYPE__GPU) {
        bool is_supported;
        rc = yaksuri_seq_pup_is_supported(type, &is_supported);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (!is_supported) {
            rc = YAKSA_ERR__NOT_SUPPORTED;
        } else {
            if (reqpriv->optype == YAKSURI_OPTYPE__PACK) {
                rc = yaksuri_seq_ipack(inbuf, outbuf, count, type, info);
                YAKSU_ERR_CHECK(rc, fn_fail);
            } else {
                rc = yaksuri_seq_iunpack(inbuf, outbuf, count, type, info);
                YAKSU_ERR_CHECK(rc, fn_fail);
            }
        }
        goto fn_exit;
    }

    /* if this cannot be handled by the CPU, queue it up for the GPU
     * to handle */
    assert(inbuf_gpudriver != YAKSURI_GPUDRIVER_ID__UNSET ||
           outbuf_gpudriver != YAKSURI_GPUDRIVER_ID__UNSET);

    if (inbuf_gpudriver != YAKSURI_GPUDRIVER_ID__UNSET &&
        outbuf_gpudriver != YAKSURI_GPUDRIVER_ID__UNSET) {
        assert(inbuf_gpudriver == outbuf_gpudriver);
        reqpriv->gpudriver_id = inbuf_gpudriver;
    } else if (inbuf_gpudriver != YAKSURI_GPUDRIVER_ID__UNSET) {
        reqpriv->gpudriver_id = inbuf_gpudriver;
    } else if (outbuf_gpudriver != YAKSURI_GPUDRIVER_ID__UNSET) {
        reqpriv->gpudriver_id = outbuf_gpudriver;
    }

    rc = yaksuri_progress_enqueue(inbuf, outbuf, count, type, info, request);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_ipack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                 yaksi_info_s * info, yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_request_s *reqpriv = (yaksuri_request_s *) request->backend.priv;

    reqpriv->optype = YAKSURI_OPTYPE__PACK;
    rc = ipup(inbuf, outbuf, count, type, info, request);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_iunpack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                   yaksi_info_s * info, yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_request_s *reqpriv = (yaksuri_request_s *) request->backend.priv;

    reqpriv->optype = YAKSURI_OPTYPE__UNPACK;
    rc = ipup(inbuf, outbuf, count, type, info, request);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
