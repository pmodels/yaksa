/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksuri_cuda.h"
#include "yaksuri_cudai.h"
#include <assert.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CHUNK_SIZE    (16)
#define MAX_PUP_BUFS  (16)

static void *malloc_fn(uintptr_t size)
{
    void *ptr = NULL;

    cudaError_t cerr = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);

    return ptr;
}

static void free_fn(void *ptr)
{
    cudaError_t cerr = cudaFree(ptr);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

yaksuri_cudai_global_s yaksuri_cudai_global;

int yaksuri_cuda_init_hook(void)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksu_pool_alloc(YAKSURI_CUDAI_PUP_BUF_SIZE, CHUNK_SIZE, MAX_PUP_BUFS, malloc_fn, free_fn,
                          &yaksuri_cudai_global.pup_buf_pool);
    YAKSU_ERR_CHECK(rc, fn_fail);

    cudaError_t cerr =
        cudaStreamCreateWithFlags(&yaksuri_cudai_global.stream, cudaStreamNonBlocking);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cuda_finalize_hook(void)
{
    int rc = YAKSA_SUCCESS;

    cudaError_t cerr = cudaStreamDestroy(yaksuri_cudai_global.stream);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    rc = yaksu_pool_free(yaksuri_cudai_global.pup_buf_pool);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
