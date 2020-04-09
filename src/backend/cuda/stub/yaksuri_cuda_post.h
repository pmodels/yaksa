/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_CUDA_POST_H_INCLUDED
#define YAKSURI_CUDA_POST_H_INCLUDED

static int yaksuri_cuda_init_hook(yaksur_gpudev_info_s ** info) ATTRIBUTE((unused));
static int yaksuri_cuda_init_hook(yaksur_gpudev_info_s ** info)
{
    *info = NULL;

    return YAKSA_SUCCESS;
}

static int yaksuri_cudai_finalize_hook(void) ATTRIBUTE((unused));
static int yaksuri_cudai_finalize_hook(void)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cudai_type_create_hook(yaksi_type_s * type, yaksur_gpudev_pup_fn * pack,
                                          yaksur_gpudev_pup_fn * unpack) ATTRIBUTE((unused));
static int yaksuri_cudai_type_create_hook(yaksi_type_s * type, yaksur_gpudev_pup_fn * pack,
                                          yaksur_gpudev_pup_fn * unpack)
{
    *pack = NULL;
    *unpack = NULL;
    return YAKSA_SUCCESS;
}

static int yaksuri_cudai_type_free_hook(yaksi_type_s * type) ATTRIBUTE((unused));
static int yaksuri_cudai_type_free_hook(yaksi_type_s * type)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cudai_event_create(void **event) ATTRIBUTE((unused));
static int yaksuri_cudai_event_create(void **event)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cudai_event_destroy(void *event) ATTRIBUTE((unused));
static int yaksuri_cudai_event_destroy(void *event)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cudai_event_query(void *event, int *completed) ATTRIBUTE((unused));
static int yaksuri_cudai_event_query(void *event, int *completed)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cudai_event_synchronize(void *event) ATTRIBUTE((unused));
static int yaksuri_cudai_event_synchronize(void *event)
{
    return YAKSA_SUCCESS;
}

static int yaksuri_cudai_get_ptr_attr(const void *buf,
                                      yaksur_ptr_attr_s * ptrattr) ATTRIBUTE((unused));
static int yaksuri_cudai_get_ptr_attr(const void *buf, yaksur_ptr_attr_s * ptrattr)
{
    ptrattr->type = YAKSUR_PTR_TYPE__UNREGISTERED_HOST;

    return YAKSA_SUCCESS;
}

#endif /* YAKSURI_CUDA_POST_H_INCLUDED */
