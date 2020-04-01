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

#ifdef HAVE_CUDA
    int inbuf_is_on_gpu, outbuf_is_on_gpu;

    rc = yaksuri_cuda_is_gpu_memory((const char *) inbuf + type->true_lb, &inbuf_is_on_gpu);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksuri_cuda_is_gpu_memory(outbuf, &outbuf_is_on_gpu);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (inbuf_is_on_gpu && outbuf_is_on_gpu) {
        if (type->backend_priv.cuda.pack) {
            rc = type->backend_priv.cuda.pack(inbuf, outbuf, count, type, request);
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

    if (type->backend_priv.seq.pack) {
        rc = type->backend_priv.seq.pack(inbuf, outbuf, count, type, request);
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
                   yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

#ifdef HAVE_CUDA
    int inbuf_is_on_gpu, outbuf_is_on_gpu;

    rc = yaksuri_cuda_is_gpu_memory(inbuf, &inbuf_is_on_gpu);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksuri_cuda_is_gpu_memory((char *) outbuf + type->true_lb, &outbuf_is_on_gpu);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (inbuf_is_on_gpu && outbuf_is_on_gpu) {
        if (type->backend_priv.cuda.unpack) {
            rc = type->backend_priv.cuda.unpack(inbuf, outbuf, count, type, request);
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

    if (type->backend_priv.seq.unpack) {
        rc = type->backend_priv.seq.unpack(inbuf, outbuf, count, type, request);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        rc = YAKSA_ERR__NOT_SUPPORTED;
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_request_test(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_seq_request_test(request);
    YAKSU_ERR_CHECK(rc, fn_fail);

#ifdef HAVE_CUDA
    rc = yaksuri_cuda_request_test(request);
    YAKSU_ERR_CHECK(rc, fn_fail);
#endif /* HAVE_CUDA */

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_request_wait(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

    while (yaksu_atomic_load(&request->cc)) {
        rc = yaksuri_seq_request_test(request);
        YAKSU_ERR_CHECK(rc, fn_fail);

#ifdef HAVE_CUDA
        rc = yaksuri_cuda_request_test(request);
        YAKSU_ERR_CHECK(rc, fn_fail);
#endif /* HAVE_CUDA */
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
