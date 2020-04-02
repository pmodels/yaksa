/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_CUDA_POST_H_INCLUDED
#define YAKSURI_CUDA_POST_H_INCLUDED

static int yaksuri_cuda_init_hook(void) ATTRIBUTE((unused));
static int yaksuri_cuda_init_hook(void)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cuda_finalize_hook(void) ATTRIBUTE((unused));
static int yaksuri_cuda_finalize_hook(void)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cuda_type_create_hook(yaksi_type_s * type) ATTRIBUTE((unused));
static int yaksuri_cuda_type_create_hook(yaksi_type_s * type)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cuda_type_free_hook(yaksi_type_s * type) ATTRIBUTE((unused));
static int yaksuri_cuda_type_free_hook(yaksi_type_s * type)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cuda_request_create_hook(yaksi_request_s * request) ATTRIBUTE((unused));
static int yaksuri_cuda_request_create_hook(yaksi_request_s * request)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cuda_request_free_hook(yaksi_request_s * request) ATTRIBUTE((unused));
static int yaksuri_cuda_request_free_hook(yaksi_request_s * request)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cuda_request_test(yaksi_request_s * request) ATTRIBUTE((unused));
static int yaksuri_cuda_request_test(yaksi_request_s * request)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cuda_request_wait(yaksi_request_s * request) ATTRIBUTE((unused));
static int yaksuri_cuda_request_wait(yaksi_request_s * request)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cuda_get_memory_type(const void *buf,
                                        yaksur_memory_type_e * memtype) ATTRIBUTE((unused));
static int yaksuri_cuda_get_memory_type(const void *buf, yaksur_memory_type_e * memtype)
{
    *memtype = YAKSUR_MEMORY_TYPE__UNREGISTERED_HOST;

    return YAKSA_SUCCESS;
}

#endif /* YAKSURI_CUDA_POST_H_INCLUDED */
