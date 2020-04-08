/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_CUDA_POST_H_INCLUDED
#define YAKSURI_CUDA_POST_H_INCLUDED

int yaksuri_cuda_init_hook(yaksu_malloc_fn * host_malloc_fn, yaksu_free_fn * host_free_fn,
                           yaksu_malloc_fn * device_malloc_fn, yaksu_free_fn * device_free_fn);
int yaksuri_cuda_finalize_hook(void);
int yaksuri_cuda_type_create_hook(yaksi_type_s * type, yaksur_gpudev_pup_fn * pack,
                                  yaksur_gpudev_pup_fn * unpack);
int yaksuri_cuda_type_free_hook(yaksi_type_s * type);

int yaksuri_cuda_event_create(void **event);
int yaksuri_cuda_event_destroy(void *event);
int yaksuri_cuda_event_query(void *event, int *completed);
int yaksuri_cuda_event_synchronize(void *event);

int yaksuri_cuda_get_memory_type(const void *buf, yaksur_memory_type_e * memtype);

#endif /* YAKSURI_CUDA_H_INCLUDED */
