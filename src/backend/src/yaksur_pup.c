/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <assert.h>
#include "yaksa.h"
#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri.h"

static int get_memory_type(const void *buf, yaksur_memory_type_e * memtype,
                           yaksuri_gpudev_id_e * id)
{
    int rc = YAKSA_SUCCESS;

    /* Each GPU backend can claim "ownership" of the input buffer */
    for (*id = YAKSURI_GPUDEV_ID__UNSET + 1; *id < YAKSURI_GPUDEV_ID__LAST; (*id)++) {
        if (yaksuri_global.gpudev[*id].info) {
            rc = yaksuri_global.gpudev[*id].info->get_memory_type(buf, memtype);
            YAKSU_ERR_CHECK(rc, fn_fail);

            if (*memtype == YAKSUR_MEMORY_TYPE__DEVICE || YAKSUR_MEMORY_TYPE__REGISTERED_HOST)
                break;
        }
    }

    if (*id == YAKSURI_GPUDEV_ID__LAST) {
        *id = YAKSURI_GPUDEV_ID__UNSET;
        *memtype = YAKSUR_MEMORY_TYPE__UNREGISTERED_HOST;
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
    yaksur_memory_type_e inbuf_memtype, outbuf_memtype;
    yaksuri_gpudev_id_e inbuf_gpudev, outbuf_gpudev, id;
    yaksuri_type_s *type_backend = (yaksuri_type_s *) type->backend.priv;
    yaksuri_request_s *request_backend = (yaksuri_request_s *) request->backend.priv;

    rc = get_memory_type((const char *) inbuf + type->true_lb, &inbuf_memtype, &inbuf_gpudev);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = get_memory_type(outbuf, &outbuf_memtype, &outbuf_gpudev);
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

    if (inbuf_memtype != YAKSUR_MEMORY_TYPE__DEVICE && outbuf_memtype != YAKSUR_MEMORY_TYPE__DEVICE) {
        if (type_backend->seq.pack) {
            rc = type_backend->seq.pack(inbuf, outbuf, count, type);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else {
            rc = YAKSA_ERR__NOT_SUPPORTED;
        }
        goto fn_exit;
    }

    request_backend->gpudev_id = id;
    assert(yaksuri_global.gpudev[id].info);
    rc = yaksuri_global.gpudev[id].info->event_create(&request_backend->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (inbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE && outbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE) {
        if (type_backend->gpudev[id].pack) {
            rc = type_backend->gpudev[id].pack(inbuf, outbuf, count, type, NULL,
                                               request_backend->event);
            YAKSU_ERR_CHECK(rc, fn_fail);

            int completed;
            rc = yaksuri_global.gpudev[id].info->event_query(request_backend->event, &completed);
            YAKSU_ERR_CHECK(rc, fn_fail);

            if (!completed) {
                yaksu_atomic_store(&request->cc, 1);
            }

            request_backend->kind = YAKSURI_REQUEST_KIND__DIRECT;
        } else {
            rc = YAKSA_ERR__NOT_SUPPORTED;
        }
    } else {
        request_backend->kind = YAKSURI_REQUEST_KIND__STAGED;

        if (inbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE &&
            outbuf_memtype == YAKSUR_MEMORY_TYPE__REGISTERED_HOST) {
            rc = yaksuri_progress_enqueue(inbuf, outbuf, count, type, request,
                                          YAKSURI_PROGRESS_ELEM_KIND__PACK_D2RH);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if (inbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE &&
                   outbuf_memtype == YAKSUR_MEMORY_TYPE__UNREGISTERED_HOST) {
            rc = yaksuri_progress_enqueue(inbuf, outbuf, count, type, request,
                                          YAKSURI_PROGRESS_ELEM_KIND__PACK_D2URH);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if (inbuf_memtype == YAKSUR_MEMORY_TYPE__REGISTERED_HOST &&
                   outbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE) {
            rc = yaksuri_progress_enqueue(inbuf, outbuf, count, type, request,
                                          YAKSURI_PROGRESS_ELEM_KIND__PACK_RH2D);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if (inbuf_memtype == YAKSUR_MEMORY_TYPE__UNREGISTERED_HOST &&
                   outbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE) {
            rc = yaksuri_progress_enqueue(inbuf, outbuf, count, type, request,
                                          YAKSURI_PROGRESS_ELEM_KIND__PACK_URH2D);
            YAKSU_ERR_CHECK(rc, fn_fail);
        }

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
    yaksur_memory_type_e inbuf_memtype, outbuf_memtype;
    yaksuri_gpudev_id_e inbuf_gpudev, outbuf_gpudev, id;
    yaksuri_type_s *type_backend = (yaksuri_type_s *) type->backend.priv;
    yaksuri_request_s *request_backend = (yaksuri_request_s *) request->backend.priv;

    rc = get_memory_type(inbuf, &inbuf_memtype, &inbuf_gpudev);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = get_memory_type((char *) outbuf + type->true_lb, &outbuf_memtype, &outbuf_gpudev);
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

    if (inbuf_memtype != YAKSUR_MEMORY_TYPE__DEVICE && outbuf_memtype != YAKSUR_MEMORY_TYPE__DEVICE) {
        if (type_backend->seq.unpack) {
            rc = type_backend->seq.unpack(inbuf, outbuf, count, type);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else {
            rc = YAKSA_ERR__NOT_SUPPORTED;
        }
        goto fn_exit;
    }

    request_backend->gpudev_id = id;
    assert(yaksuri_global.gpudev[id].info);
    rc = yaksuri_global.gpudev[id].info->event_create(&request_backend->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (inbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE && outbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE) {
        if (type_backend->gpudev[id].unpack) {
            rc = type_backend->gpudev[id].unpack(inbuf, outbuf, count, type, NULL,
                                                 request_backend->event);
            YAKSU_ERR_CHECK(rc, fn_fail);

            int completed;
            rc = yaksuri_global.gpudev[id].info->event_query(request_backend->event, &completed);
            YAKSU_ERR_CHECK(rc, fn_fail);

            if (!completed) {
                yaksu_atomic_store(&request->cc, 1);
            }

            request_backend->kind = YAKSURI_REQUEST_KIND__DIRECT;
        } else {
            rc = YAKSA_ERR__NOT_SUPPORTED;
        }
    } else {
        request_backend->kind = YAKSURI_REQUEST_KIND__STAGED;

        if (inbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE &&
            outbuf_memtype == YAKSUR_MEMORY_TYPE__REGISTERED_HOST) {
            rc = yaksuri_progress_enqueue(inbuf, outbuf, count, type, request,
                                          YAKSURI_PROGRESS_ELEM_KIND__UNPACK_D2RH);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if (inbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE &&
                   outbuf_memtype == YAKSUR_MEMORY_TYPE__UNREGISTERED_HOST) {
            rc = yaksuri_progress_enqueue(inbuf, outbuf, count, type, request,
                                          YAKSURI_PROGRESS_ELEM_KIND__UNPACK_D2URH);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if (inbuf_memtype == YAKSUR_MEMORY_TYPE__REGISTERED_HOST &&
                   outbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE) {
            rc = yaksuri_progress_enqueue(inbuf, outbuf, count, type, request,
                                          YAKSURI_PROGRESS_ELEM_KIND__UNPACK_RH2D);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if (inbuf_memtype == YAKSUR_MEMORY_TYPE__UNREGISTERED_HOST &&
                   outbuf_memtype == YAKSUR_MEMORY_TYPE__DEVICE) {
            rc = yaksuri_progress_enqueue(inbuf, outbuf, count, type, request,
                                          YAKSURI_PROGRESS_ELEM_KIND__UNPACK_URH2D);
            YAKSU_ERR_CHECK(rc, fn_fail);
        }

        rc = yaksuri_progress_poke();
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
