/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSU_POOL_H_INCLUDED
#define YAKSU_POOL_H_INCLUDED

typedef void *yaksu_pool_s;

typedef void *(*yaksu_pool_malloc_fn) (uintptr_t);
typedef void (*yaksu_pool_free_fn) (void *);

int yaksu_pool_alloc(uintptr_t elemsize, uintptr_t elems_in_chunk, uintptr_t maxelems,
                     yaksu_pool_malloc_fn malloc_fn, yaksu_pool_free_fn free_fn,
                     yaksu_pool_s * pool);
int yaksu_pool_free(yaksu_pool_s pool);
int yaksu_pool_elem_alloc(yaksu_pool_s pool, void **elem, int *elem_idx);
int yaksu_pool_elem_free(yaksu_pool_s pool, int idx);
int yaksu_pool_elem_get(yaksu_pool_s pool, int elem_idx, void **elem);

#endif /* YAKSU_POOL_H_INCLUDED */
