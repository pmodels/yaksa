/*
* Copyright (C) by Argonne National Laboratory
*     See COPYRIGHT in top-level directory
*/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "yaksi.h"
#include "yaksuri_cudai.h"

int yaksuri_cudai_pup_is_supported(yaksi_type_s * type, bool * is_supported)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_type_s *cuda_type = (yaksuri_cudai_type_s *) type->backend.cuda.priv;

    if (type->kind != YAKSI_TYPE_KIND__BUILTIN && !cuda_type->pack)
        *is_supported = false;
    else
        *is_supported = true;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cudai_ipack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                        void *device_tmpbuf, void *event)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_type_s *cuda_type = (yaksuri_cudai_type_s *) type->backend.cuda.priv;
    cudaError_t cerr;

    /* shortcut for builtin types */
    if (type->kind == YAKSI_TYPE_KIND__BUILTIN) {
        cerr = cudaMemcpyAsync(outbuf, inbuf, count * type->size, cudaMemcpyDefault,
                               yaksuri_cudai_global.stream);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
    } else if (!cuda_type->pack) {
        rc = YAKSA_ERR__NOT_SUPPORTED;
        goto fn_exit;
    } else {
        struct cudaPointerAttributes outattr, inattr;

        cerr = cudaPointerGetAttributes(&outattr, outbuf);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

        cerr = cudaPointerGetAttributes(&inattr, (char *) inbuf + type->true_lb);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

        rc = yaksuri_cudai_md_alloc(type);
        YAKSU_ERR_CHECK(rc, fn_fail);

        int n_threads = YAKSURI_CUDAI_THREAD_BLOCK_SIZE;
        int n_blocks = count * cuda_type->num_elements / YAKSURI_CUDAI_THREAD_BLOCK_SIZE;
        n_blocks += ! !(count * cuda_type->num_elements % YAKSURI_CUDAI_THREAD_BLOCK_SIZE);

        if (outattr.type != cudaMemoryTypeDevice && outattr.type != cudaMemoryTypeManaged) {
            cuda_type->pack(inbuf, device_tmpbuf, count, cuda_type->md, n_threads, n_blocks);
            cerr = cudaMemcpyAsync(outbuf, device_tmpbuf, count * type->size, cudaMemcpyDefault,
                                   yaksuri_cudai_global.stream);
            YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
        } else {
            cuda_type->pack(inbuf, outbuf, count, cuda_type->md, n_threads, n_blocks);
        }
    }

    cerr = cudaEventRecord((cudaEvent_t) event, yaksuri_cudai_global.stream);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cudai_iunpack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                          void *device_tmpbuf, void *event)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_type_s *cuda_type = (yaksuri_cudai_type_s *) type->backend.cuda.priv;
    cudaError_t cerr;

    /* shortcut for builtin types */
    if (type->kind == YAKSI_TYPE_KIND__BUILTIN) {
        cerr =
            cudaMemcpyAsync(outbuf, inbuf, count * type->size, cudaMemcpyDefault,
                            yaksuri_cudai_global.stream);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
    } else if (!cuda_type->unpack) {
        rc = YAKSA_ERR__NOT_SUPPORTED;
        goto fn_exit;
    } else {
        struct cudaPointerAttributes outattr, inattr;

        cerr = cudaPointerGetAttributes(&outattr, (char *) outbuf + type->true_lb);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

        cerr = cudaPointerGetAttributes(&inattr, inbuf);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

        rc = yaksuri_cudai_md_alloc(type);
        YAKSU_ERR_CHECK(rc, fn_fail);

        int n_threads = YAKSURI_CUDAI_THREAD_BLOCK_SIZE;
        int n_blocks = count * cuda_type->num_elements / YAKSURI_CUDAI_THREAD_BLOCK_SIZE;
        n_blocks += ! !(count * cuda_type->num_elements % YAKSURI_CUDAI_THREAD_BLOCK_SIZE);

        if (inattr.type != cudaMemoryTypeDevice && inattr.type != cudaMemoryTypeManaged) {
            cerr = cudaMemcpyAsync(device_tmpbuf, inbuf, count * type->size, cudaMemcpyDefault,
                                   yaksuri_cudai_global.stream);
            YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
            cuda_type->unpack(device_tmpbuf, outbuf, count, cuda_type->md, n_threads, n_blocks);
        } else {
            cuda_type->unpack(inbuf, outbuf, count, cuda_type->md, n_threads, n_blocks);
        }
    }

    cerr = cudaEventRecord((cudaEvent_t) event, yaksuri_cudai_global.stream);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
