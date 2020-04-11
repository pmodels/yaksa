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

#endif /* YAKSURI_CUDA_POST_H_INCLUDED */
