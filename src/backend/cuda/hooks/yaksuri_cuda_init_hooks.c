/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksuri_cudai.h"
#include <assert.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

static void *cuda_host_malloc(uintptr_t size)
{
    void *ptr = NULL;

    cudaError_t cerr = cudaMallocHost(&ptr, size);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);

    return ptr;
}

static void *cuda_device_malloc(uintptr_t size)
{
    void *ptr = NULL;

    cudaError_t cerr = cudaMalloc(&ptr, size);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);

    return ptr;
}

static void cuda_host_free(void *ptr)
{
    cudaError_t cerr = cudaFreeHost(ptr);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

static void cuda_device_free(void *ptr)
{
    cudaError_t cerr = cudaFree(ptr);
    YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr);
}

yaksuri_cudai_global_s yaksuri_cudai_global;

static int finalize_hook(void)
{
    int rc = YAKSA_SUCCESS;

    cudaError_t cerr = cudaStreamDestroy(yaksuri_cudai_global.stream);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cuda_init_hook(yaksur_gpudev_info_s ** info)
{
    int rc = YAKSA_SUCCESS;

    *info = (yaksur_gpudev_info_s *) malloc(sizeof(yaksur_gpudev_info_s));

    (*info)->ipack = yaksuri_cudai_ipack;
    (*info)->iunpack = yaksuri_cudai_iunpack;
    (*info)->pup_is_supported = yaksuri_cudai_pup_is_supported;
    (*info)->host_malloc = cuda_host_malloc;
    (*info)->host_free = cuda_host_free;
    (*info)->device_malloc = cuda_device_malloc;
    (*info)->device_free = cuda_device_free;
    (*info)->event_destroy = yaksuri_cudai_event_destroy;
    (*info)->event_query = yaksuri_cudai_event_query;
    (*info)->event_synchronize = yaksuri_cudai_event_synchronize;
    (*info)->type_create = yaksuri_cudai_type_create_hook;
    (*info)->type_free = yaksuri_cudai_type_free_hook;
    (*info)->get_ptr_attr = yaksuri_cudai_get_ptr_attr;
    (*info)->finalize = finalize_hook;

    cudaError_t cerr =
        cudaStreamCreateWithFlags(&yaksuri_cudai_global.stream, cudaStreamNonBlocking);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
