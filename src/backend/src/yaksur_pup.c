/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <assert.h>
#include "yaksa.h"
#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri.h"

static int get_ptr_attr(const void *buf, yaksur_ptr_attr_s * ptrattr, yaksuri_gpudev_id_e * id)
{
    int rc = YAKSA_SUCCESS;

    /* Each GPU backend can claim "ownership" of the input buffer */
    for (*id = YAKSURI_GPUDEV_ID__UNSET + 1; *id < YAKSURI_GPUDEV_ID__LAST; (*id)++) {
        if (yaksuri_global.gpudev[*id].info) {
            rc = yaksuri_global.gpudev[*id].info->get_ptr_attr(buf, ptrattr);
            YAKSU_ERR_CHECK(rc, fn_fail);

            if (ptrattr->type == YAKSUR_PTR_TYPE__DEVICE ||
                ptrattr->type == YAKSUR_PTR_TYPE__REGISTERED_HOST)
                break;
        }
    }

    if (*id == YAKSURI_GPUDEV_ID__LAST) {
        *id = YAKSURI_GPUDEV_ID__UNSET;
        ptrattr->type = YAKSUR_PTR_TYPE__UNREGISTERED_HOST;
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_ipack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                 yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;
    yaksur_ptr_attr_s inattr, outattr;
    yaksuri_gpudev_id_e inbuf_gpudev, outbuf_gpudev, id;
    yaksuri_request_s *request_backend = (yaksuri_request_s *) request->backend.priv;

    rc = get_ptr_attr((const char *) inbuf + type->true_lb, &inattr, &inbuf_gpudev);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = get_ptr_attr(outbuf, &outattr, &outbuf_gpudev);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (inbuf_gpudev == YAKSURI_GPUDEV_ID__UNSET && outbuf_gpudev == YAKSURI_GPUDEV_ID__UNSET) {
        id = YAKSURI_GPUDEV_ID__UNSET;
    } else if (inbuf_gpudev != YAKSURI_GPUDEV_ID__UNSET &&
               outbuf_gpudev != YAKSURI_GPUDEV_ID__UNSET) {
        assert(inbuf_gpudev == outbuf_gpudev);
        id = inbuf_gpudev;
    } else if (inbuf_gpudev != YAKSURI_GPUDEV_ID__UNSET) {
        id = inbuf_gpudev;
    } else if (outbuf_gpudev != YAKSURI_GPUDEV_ID__UNSET) {
        id = outbuf_gpudev;
    }


    /* if this can be handled by the CPU, wrap it up */
    if (inattr.type != YAKSUR_PTR_TYPE__DEVICE && outattr.type != YAKSUR_PTR_TYPE__DEVICE) {
        bool is_supported;
        rc = yaksuri_seq_pup_is_supported(type, &is_supported);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (!is_supported) {
            rc = YAKSA_ERR__NOT_SUPPORTED;
        } else {
            rc = yaksuri_seq_ipack(inbuf, outbuf, count, type);
            YAKSU_ERR_CHECK(rc, fn_fail);
        }
        goto fn_exit;
    }


    /* if the GPU backend cannot support this type, return */
    bool is_supported;
    rc = yaksuri_global.gpudev[id].info->pup_is_supported(type, &is_supported);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (!is_supported) {
        rc = YAKSA_ERR__NOT_SUPPORTED;
        goto fn_exit;
    }


    request_backend->gpudev_id = id;
    assert(yaksuri_global.gpudev[id].info);

    if (inattr.type == YAKSUR_PTR_TYPE__DEVICE && outattr.type == YAKSUR_PTR_TYPE__DEVICE &&
        inattr.device == outattr.device) {
        /* if the GPU can handle the data movement without any
         * temporary buffers, wrap it up */
        rc = yaksuri_global.gpudev[id].info->ipack(inbuf, outbuf, count, type, NULL,
                                                   NULL, &request_backend->event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        int completed;
        rc = yaksuri_global.gpudev[id].info->event_query(request_backend->event, &completed);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (!completed) {
            yaksu_atomic_store(&request->cc, 1);
        }

        request_backend->kind = YAKSURI_REQUEST_KIND__DIRECT;
    } else {
        /* we need temporary buffers and pipelining; queue it up in
         * the progress engine */
        request_backend->kind = YAKSURI_REQUEST_KIND__STAGED;

        rc = yaksuri_progress_enqueue(inbuf, outbuf, count, type, request,
                                      inattr, outattr, YAKSURI_PUPTYPE__PACK);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = yaksuri_progress_poke();
        YAKSU_ERR_CHECK(rc, fn_fail);
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
    yaksur_ptr_attr_s inattr, outattr;
    yaksuri_gpudev_id_e inbuf_gpudev, outbuf_gpudev, id;
    yaksuri_request_s *request_backend = (yaksuri_request_s *) request->backend.priv;

    rc = get_ptr_attr(inbuf, &inattr, &inbuf_gpudev);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = get_ptr_attr((char *) outbuf + type->true_lb, &outattr, &outbuf_gpudev);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (inbuf_gpudev == YAKSURI_GPUDEV_ID__UNSET && outbuf_gpudev == YAKSURI_GPUDEV_ID__UNSET) {
        id = YAKSURI_GPUDEV_ID__UNSET;
    } else if (inbuf_gpudev != YAKSURI_GPUDEV_ID__UNSET &&
               outbuf_gpudev != YAKSURI_GPUDEV_ID__UNSET) {
        assert(inbuf_gpudev == outbuf_gpudev);
        id = inbuf_gpudev;
    } else if (inbuf_gpudev != YAKSURI_GPUDEV_ID__UNSET) {
        id = inbuf_gpudev;
    } else if (outbuf_gpudev != YAKSURI_GPUDEV_ID__UNSET) {
        id = outbuf_gpudev;
    }


    /* if this can be handled by the CPU, wrap it up */
    if (inattr.type != YAKSUR_PTR_TYPE__DEVICE && outattr.type != YAKSUR_PTR_TYPE__DEVICE) {
        bool is_supported;
        rc = yaksuri_seq_pup_is_supported(type, &is_supported);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (!is_supported) {
            rc = YAKSA_ERR__NOT_SUPPORTED;
        } else {
            rc = yaksuri_seq_iunpack(inbuf, outbuf, count, type);
            YAKSU_ERR_CHECK(rc, fn_fail);
        }
        goto fn_exit;
    }


    /* if the GPU backend cannot support this type, return */
    bool is_supported;
    rc = yaksuri_global.gpudev[id].info->pup_is_supported(type, &is_supported);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (!is_supported) {
        rc = YAKSA_ERR__NOT_SUPPORTED;
        goto fn_exit;
    }


    request_backend->gpudev_id = id;
    assert(yaksuri_global.gpudev[id].info);

    if (inattr.type == YAKSUR_PTR_TYPE__DEVICE && outattr.type == YAKSUR_PTR_TYPE__DEVICE &&
        inattr.device == outattr.device) {
        /* if the GPU can handle the data movement without any
         * temporary buffers, wrap it up */
        rc = yaksuri_global.gpudev[id].info->iunpack(inbuf, outbuf, count, type, NULL,
                                                     NULL, &request_backend->event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        int completed;
        rc = yaksuri_global.gpudev[id].info->event_query(request_backend->event, &completed);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (!completed) {
            yaksu_atomic_store(&request->cc, 1);
        }

        request_backend->kind = YAKSURI_REQUEST_KIND__DIRECT;
    } else {
        /* we need temporary buffers and pipelining; queue it up in
         * the progress engine */
        request_backend->kind = YAKSURI_REQUEST_KIND__STAGED;

        rc = yaksuri_progress_enqueue(inbuf, outbuf, count, type, request,
                                      inattr, outattr, YAKSURI_PUPTYPE__UNPACK);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = yaksuri_progress_poke();
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
