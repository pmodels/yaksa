/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSU_BUFFER_POOL_H_INCLUDED
#define YAKSU_BUFFER_POOL_H_INCLUDED

typedef void *yaksu_buffer_pool_s;

typedef void *(*yaksu_malloc_fn) (uintptr_t);
typedef void (*yaksu_free_fn) (void *);

int yaksu_buffer_pool_alloc(uintptr_t elemsize, unsigned int elems_in_chunk, unsigned int maxelems,
                            yaksu_malloc_fn malloc_fn, yaksu_free_fn free_fn,
                            yaksu_buffer_pool_s * pool);
int yaksu_buffer_pool_free(yaksu_buffer_pool_s pool);
int yaksu_buffer_pool_elem_alloc(yaksu_buffer_pool_s pool, void **elem, unsigned int *elem_idx);
int yaksu_buffer_pool_elem_free(yaksu_buffer_pool_s pool, unsigned int idx);
int yaksu_buffer_pool_elem_get(yaksu_buffer_pool_s pool, unsigned int elem_idx, void **elem);

#endif /* YAKSU_BUFFER_POOL_H_INCLUDED */
