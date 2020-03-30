/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksa.h"
#include "yaksi.h"
#include "yaksu.h"
#include "yaksur.h"
#include "yaksuri.h"

int yaksur_ipack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                 yaksi_request_s ** request)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_type_s *backend = (yaksuri_type_s *) type->backend;

#ifdef HAVE_CUDA
    int inbuf_is_on_gpu, outbuf_is_on_gpu;

    rc = yaksuri_cuda_is_gpu_memory((const char *) inbuf + type->true_lb, &inbuf_is_on_gpu);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksuri_cuda_is_gpu_memory(outbuf, &outbuf_is_on_gpu);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (inbuf_is_on_gpu && outbuf_is_on_gpu) {
        if (backend->cuda.pack) {
            rc = backend->cuda.pack(inbuf, outbuf, count, type, request);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else {
            rc = YAKSA_ERR__NOT_SUPPORTED;
        }
        goto fn_exit;
    } else if (inbuf_is_on_gpu != outbuf_is_on_gpu) {
        /* FIXME: we need to support this case */
        rc = YAKSA_ERR__NOT_SUPPORTED;
        goto fn_exit;
    }
#endif

    if (backend->seq.pack) {
        rc = backend->seq.pack(inbuf, outbuf, count, type, request);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        rc = YAKSA_ERR__NOT_SUPPORTED;
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_iunpack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                   yaksi_request_s ** request)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_type_s *backend = (yaksuri_type_s *) type->backend;

#ifdef HAVE_CUDA
    int inbuf_is_on_gpu, outbuf_is_on_gpu;

    rc = yaksuri_cuda_is_gpu_memory(inbuf, &inbuf_is_on_gpu);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksuri_cuda_is_gpu_memory((char *) outbuf + type->true_lb, &outbuf_is_on_gpu);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (inbuf_is_on_gpu && outbuf_is_on_gpu) {
        if (backend->cuda.unpack) {
            rc = backend->cuda.unpack(inbuf, outbuf, count, type, request);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else {
            rc = YAKSA_ERR__NOT_SUPPORTED;
        }
        goto fn_exit;
    } else if (inbuf_is_on_gpu != outbuf_is_on_gpu) {
        /* FIXME: we need to support this case */
        rc = YAKSA_ERR__NOT_SUPPORTED;
        goto fn_exit;
    }
#endif

    if (backend->seq.unpack) {
        rc = backend->seq.unpack(inbuf, outbuf, count, type, request);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        rc = YAKSA_ERR__NOT_SUPPORTED;
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
