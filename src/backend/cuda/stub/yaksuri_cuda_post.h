/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_CUDA_POST_H_INCLUDED
#define YAKSURI_CUDA_POST_H_INCLUDED

static int yaksuri_cuda_init_hook(yaksu_malloc_fn * host_malloc_fn, yaksu_free_fn * host_free_fn,
                                  yaksu_malloc_fn * device_malloc_fn,
                                  yaksu_free_fn * device_free_fn);
static int yaksuri_cuda_init_hook(yaksu_malloc_fn * host_malloc_fn, yaksu_free_fn * host_free_fn,
                                  yaksu_malloc_fn * device_malloc_fn,
                                  yaksu_free_fn * device_free_fn)
{
    *host_malloc_fn = NULL;
    *host_free_fn = NULL;
    *device_malloc_fn = NULL;
    *device_free_fn = NULL;

    return YAKSA_SUCCESS;
}

static int yaksuri_cuda_finalize_hook(void) ATTRIBUTE((unused));
static int yaksuri_cuda_finalize_hook(void)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cuda_type_create_hook(yaksi_type_s * type, yaksur_gpudev_pup_fn * pack,
                                         yaksur_gpudev_pup_fn * unpack) ATTRIBUTE((unused));
static int yaksuri_cuda_type_create_hook(yaksi_type_s * type, yaksur_gpudev_pup_fn * pack,
                                         yaksur_gpudev_pup_fn * unpack)
{
    *pack = NULL;
    *unpack = NULL;
    return YAKSA_SUCCESS;
}

static int yaksuri_cuda_type_free_hook(yaksi_type_s * type) ATTRIBUTE((unused));
static int yaksuri_cuda_type_free_hook(yaksi_type_s * type)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cuda_event_create(void **event) ATTRIBUTE((unused));
static int yaksuri_cuda_event_create(void **event)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cuda_event_destroy(void *event) ATTRIBUTE((unused));
static int yaksuri_cuda_event_destroy(void *event)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cuda_event_query(void *event, int *completed) ATTRIBUTE((unused));
static int yaksuri_cuda_event_query(void *event, int *completed)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cuda_event_synchronize(void *event) ATTRIBUTE((unused));
static int yaksuri_cuda_event_synchronize(void *event)
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
