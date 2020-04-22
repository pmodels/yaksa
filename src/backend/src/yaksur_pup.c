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

    /* Each GPU backend can claim "ownership" of the input buffer */
    for (*id = YAKSURI_GPUDRIVER_ID__UNSET + 1; *id < YAKSURI_GPUDRIVER_ID__LAST; (*id)++) {
        if (yaksuri_global.gpudriver[*id].info) {
            rc = yaksuri_global.gpudriver[*id].info->get_ptr_attr(buf, ptrattr);
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

/*
 * In all of the "DIRECT" cases below, there are a few important
 * things to note:
 *
 *  1. We increment the completion counter only for the first
 *     incomplete request.  Future incomplete requests simply
 *     overwrite the event.
 *
 *  2. We use an increment instead of an atomic store, because some
 *     operations in this request might go through the progress engine
 *     (STAGED).
 */

int yaksur_ipack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                 yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;
    yaksur_ptr_attr_s inattr, outattr;
    yaksuri_gpudriver_id_e inbuf_gpudriver, outbuf_gpudriver, id;
    yaksuri_request_s *request_backend = (yaksuri_request_s *) request->backend.priv;

    rc = get_ptr_attr((const char *) inbuf + type->true_lb, &inattr, &inbuf_gpudriver);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = get_ptr_attr(outbuf, &outattr, &outbuf_gpudriver);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (inbuf_gpudriver == YAKSURI_GPUDRIVER_ID__UNSET &&
        outbuf_gpudriver == YAKSURI_GPUDRIVER_ID__UNSET) {
        id = YAKSURI_GPUDRIVER_ID__UNSET;
    } else if (inbuf_gpudriver != YAKSURI_GPUDRIVER_ID__UNSET &&
               outbuf_gpudriver != YAKSURI_GPUDRIVER_ID__UNSET) {
        assert(inbuf_gpudriver == outbuf_gpudriver);
        id = inbuf_gpudriver;
    } else if (inbuf_gpudriver != YAKSURI_GPUDRIVER_ID__UNSET) {
        id = inbuf_gpudriver;
    } else if (outbuf_gpudriver != YAKSURI_GPUDRIVER_ID__UNSET) {
        id = outbuf_gpudriver;
    }


    /* if this can be handled by the CPU, wrap it up */
    if (inattr.type != YAKSUR_PTR_TYPE__GPU && outattr.type != YAKSUR_PTR_TYPE__GPU) {
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
    rc = yaksuri_global.gpudriver[id].info->pup_is_supported(type, &is_supported);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (!is_supported) {
        rc = YAKSA_ERR__NOT_SUPPORTED;
        goto fn_exit;
    }


    request_backend->gpudriver_id = id;
    assert(yaksuri_global.gpudriver[id].info);

    if (inattr.type == YAKSUR_PTR_TYPE__GPU && outattr.type == YAKSUR_PTR_TYPE__GPU &&
        inattr.device == outattr.device) {
        /* gpu-to-gpu copies do not need temporary buffers */
        bool first_event = !request_backend->event;
        rc = yaksuri_global.gpudriver[id].info->ipack(inbuf, outbuf, count, type, NULL,
                                                      inattr.device, NULL, &request_backend->event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (first_event) {
            yaksu_atomic_incr(&request->cc);
        }

        /* if the request kind was already set to STAGED, do not
         * override it, as a part of the request could be staged */
        if (request_backend->kind == YAKSURI_REQUEST_KIND__UNSET) {
            request_backend->kind = YAKSURI_REQUEST_KIND__DIRECT;
        }
    } else if (type->is_contig && inattr.type == YAKSUR_PTR_TYPE__GPU &&
               outattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST) {
        /* gpu-to-host or host-to-gpu copies do not need
         * temporary buffers either, if the host buffer is registered
         * and the type is contiguous */
        bool first_event = !request_backend->event;
        rc = yaksuri_global.gpudriver[id].info->ipack(inbuf, outbuf, count, type, NULL,
                                                      inattr.device, NULL, &request_backend->event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (first_event) {
            yaksu_atomic_incr(&request->cc);
        }

        /* if the request kind was already set to STAGED, do not
         * override it, as a part of the request could be staged */
        if (request_backend->kind == YAKSURI_REQUEST_KIND__UNSET) {
            request_backend->kind = YAKSURI_REQUEST_KIND__DIRECT;
        }
    } else if (type->is_contig && inattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST &&
               outattr.type == YAKSUR_PTR_TYPE__GPU) {
        /* gpu-to-host or host-to-gpu copies do not need
         * temporary buffers either, if the host buffer is registered
         * and the type is contiguous */
        bool first_event = !request_backend->event;
        rc = yaksuri_global.gpudriver[id].info->ipack(inbuf, outbuf, count, type, NULL,
                                                      outattr.device, NULL,
                                                      &request_backend->event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (first_event) {
            yaksu_atomic_incr(&request->cc);
        }

        /* if the request kind was already set to STAGED, do not
         * override it, as a part of the request could be staged */
        if (request_backend->kind == YAKSURI_REQUEST_KIND__UNSET) {
            request_backend->kind = YAKSURI_REQUEST_KIND__DIRECT;
        }
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
    yaksuri_gpudriver_id_e inbuf_gpudriver, outbuf_gpudriver, id;
    yaksuri_request_s *request_backend = (yaksuri_request_s *) request->backend.priv;

    rc = get_ptr_attr(inbuf, &inattr, &inbuf_gpudriver);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = get_ptr_attr((char *) outbuf + type->true_lb, &outattr, &outbuf_gpudriver);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (inbuf_gpudriver == YAKSURI_GPUDRIVER_ID__UNSET &&
        outbuf_gpudriver == YAKSURI_GPUDRIVER_ID__UNSET) {
        id = YAKSURI_GPUDRIVER_ID__UNSET;
    } else if (inbuf_gpudriver != YAKSURI_GPUDRIVER_ID__UNSET &&
               outbuf_gpudriver != YAKSURI_GPUDRIVER_ID__UNSET) {
        assert(inbuf_gpudriver == outbuf_gpudriver);
        id = inbuf_gpudriver;
    } else if (inbuf_gpudriver != YAKSURI_GPUDRIVER_ID__UNSET) {
        id = inbuf_gpudriver;
    } else if (outbuf_gpudriver != YAKSURI_GPUDRIVER_ID__UNSET) {
        id = outbuf_gpudriver;
    }


    /* if this can be handled by the CPU, wrap it up */
    if (inattr.type != YAKSUR_PTR_TYPE__GPU && outattr.type != YAKSUR_PTR_TYPE__GPU) {
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
    rc = yaksuri_global.gpudriver[id].info->pup_is_supported(type, &is_supported);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (!is_supported) {
        rc = YAKSA_ERR__NOT_SUPPORTED;
        goto fn_exit;
    }


    request_backend->gpudriver_id = id;
    assert(yaksuri_global.gpudriver[id].info);

    if (inattr.type == YAKSUR_PTR_TYPE__GPU && outattr.type == YAKSUR_PTR_TYPE__GPU &&
        inattr.device == outattr.device) {
        /* gpu-to-gpu copies do not need temporary buffers */
        bool first_event = !request_backend->event;
        rc = yaksuri_global.gpudriver[id].info->iunpack(inbuf, outbuf, count, type, NULL,
                                                        inattr.device, NULL,
                                                        &request_backend->event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (first_event) {
            yaksu_atomic_incr(&request->cc);
        }

        /* if the request kind was already set to STAGED, do not
         * override it, as a part of the request could be staged */
        if (request_backend->kind == YAKSURI_REQUEST_KIND__UNSET) {
            request_backend->kind = YAKSURI_REQUEST_KIND__DIRECT;
        }
    } else if (type->is_contig && inattr.type == YAKSUR_PTR_TYPE__GPU &&
               outattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST) {
        /* gpu-to-host or host-to-gpu copies do not need
         * temporary buffers either, if the host buffer is registered
         * and the type is contiguous */
        bool first_event = !request_backend->event;
        rc = yaksuri_global.gpudriver[id].info->iunpack(inbuf, outbuf, count, type, NULL,
                                                        inattr.device, NULL,
                                                        &request_backend->event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (first_event) {
            yaksu_atomic_incr(&request->cc);
        }

        /* if the request kind was already set to STAGED, do not
         * override it, as a part of the request could be staged */
        if (request_backend->kind == YAKSURI_REQUEST_KIND__UNSET) {
            request_backend->kind = YAKSURI_REQUEST_KIND__DIRECT;
        }
    } else if (type->is_contig && inattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST &&
               outattr.type == YAKSUR_PTR_TYPE__GPU) {
        /* gpu-to-host or host-to-gpu copies do not need
         * temporary buffers either, if the host buffer is registered
         * and the type is contiguous */
        bool first_event = !request_backend->event;
        rc = yaksuri_global.gpudriver[id].info->iunpack(inbuf, outbuf, count, type, NULL,
                                                        outattr.device, NULL,
                                                        &request_backend->event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (first_event) {
            yaksu_atomic_incr(&request->cc);
        }

        /* if the request kind was already set to STAGED, do not
         * override it, as a part of the request could be staged */
        if (request_backend->kind == YAKSURI_REQUEST_KIND__UNSET) {
            request_backend->kind = YAKSURI_REQUEST_KIND__DIRECT;
        }
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
