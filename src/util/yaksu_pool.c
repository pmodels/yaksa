/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksa.h"
#include "yaksu.h"
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>

typedef struct chunk {
    void **elems;
    void *slab;
    struct chunk *next;
} chunk_s;

typedef struct pool_head {
    uintptr_t elemsize;
    unsigned int elems_in_chunk;
    unsigned int maxelems;

    yaksu_malloc_fn malloc_fn;
    yaksu_free_fn free_fn;

    pthread_mutex_t mutex;

    unsigned int current_num_chunks;
    unsigned int max_num_chunks;
    chunk_s *chunks;
} pool_head_s;

static pthread_mutex_t global_mutex = PTHREAD_MUTEX_INITIALIZER;

int yaksu_pool_alloc(uintptr_t elemsize, unsigned int elems_in_chunk, unsigned int maxelems,
                     yaksu_malloc_fn malloc_fn, yaksu_free_fn free_fn, yaksu_pool_s * pool)
{
    int rc = YAKSA_SUCCESS;
    pool_head_s *pool_head;

    pthread_mutex_lock(&global_mutex);

    pool_head = malloc(sizeof(pool_head_s));

    pool_head->elemsize = elemsize;
    pool_head->elems_in_chunk = elems_in_chunk;
    pool_head->maxelems = maxelems;

    pool_head->malloc_fn = malloc_fn;
    pool_head->free_fn = free_fn;

    pthread_mutex_init(&pool_head->mutex, NULL);

    pool_head->current_num_chunks = 0;
    pool_head->max_num_chunks = maxelems / elems_in_chunk;
    pool_head->chunks = NULL;

    *pool = (void *) pool_head;

    pthread_mutex_unlock(&global_mutex);
    return rc;
}

int yaksu_pool_free(yaksu_pool_s pool)
{
    int rc = YAKSA_SUCCESS;
    pool_head_s *pool_head = (pool_head_s *) pool;

    pthread_mutex_lock(&global_mutex);

    int count = 0;
    for (chunk_s * chunk = pool_head->chunks; chunk;) {
        chunk_s *next = chunk->next;
        for (unsigned int i = 0; i < pool_head->elems_in_chunk; i++)
            if (chunk->elems[i])
                count++;

        pool_head->free_fn(chunk->slab);
        free(chunk->elems);
        free(chunk);
        chunk = next;
    }

    /* free self */
    pthread_mutex_destroy(&pool_head->mutex);
    free(pool_head);

    if (count) {
        fprintf(stderr, "[WARNING] yaksa: %d leaked handles\n", count);
        fflush(stderr);
    }

    pthread_mutex_unlock(&global_mutex);
    return rc;
}

int yaksu_pool_elem_alloc(yaksu_pool_s pool, void **elem, unsigned int *elem_idx)
{
    int rc = YAKSA_SUCCESS;
    pool_head_s *pool_head = (pool_head_s *) pool;
    int idx;
    chunk_s *new = NULL;

    pthread_mutex_lock(&pool_head->mutex);

    /* try to find an available type */
    idx = 0;
    for (chunk_s * chunk = pool_head->chunks; chunk; chunk = chunk->next) {
        for (unsigned int i = 0; i < pool_head->elems_in_chunk; i++) {
            if (chunk->elems[i] == NULL) {
                chunk->elems[i] = (char *) chunk->slab + i * pool_head->elemsize;
                YAKSU_ERR_CHKANDJUMP(!chunk->elems[i], rc, YAKSA_ERR__OUT_OF_MEM, fn_fail);
                *elem_idx = idx;
                *elem = chunk->elems[i];
                goto fn_exit;
            }
            idx++;
        }
    }

    /* no empty slot available; see if we can allocate more chunks */
    assert(pool_head->current_num_chunks <= pool_head->max_num_chunks);
    if (pool_head->current_num_chunks == pool_head->max_num_chunks)
        goto fn_exit;

    /* allocate another chunk */
    idx = 0;
    new = (chunk_s *) malloc(sizeof(chunk_s));
    YAKSU_ERR_CHKANDJUMP(!new, rc, YAKSA_ERR__OUT_OF_MEM, fn_fail);

    new->slab = pool_head->malloc_fn(pool_head->elems_in_chunk * pool_head->elemsize);
    YAKSU_ERR_CHKANDJUMP(!new->slab, rc, YAKSA_ERR__OUT_OF_MEM, fn_fail);

    new->elems = (void **) malloc(pool_head->elems_in_chunk * sizeof(void *));
    YAKSU_ERR_CHKANDJUMP(!new->elems, rc, YAKSA_ERR__OUT_OF_MEM, fn_fail);

    memset(new->elems, 0, pool_head->elems_in_chunk * sizeof(void *));
    new->next = NULL;

    pool_head->current_num_chunks++;

    if (pool_head->chunks == NULL) {
        pool_head->chunks = new;
    } else {
        chunk_s *chunk;
        idx += pool_head->elems_in_chunk;
        for (chunk = pool_head->chunks; chunk->next; chunk = chunk->next) {
            idx += pool_head->elems_in_chunk;
        }
        chunk->next = new;
    }

    new->elems[0] = new->slab;
    *elem_idx = idx;
    *elem = new->elems[0];

  fn_exit:
    pthread_mutex_unlock(&pool_head->mutex);
    return rc;
  fn_fail:
    if (new) {
        free(new->elems);
        free(new->slab);
        free(new);
    }
    goto fn_exit;
}

int yaksu_pool_elem_free(yaksu_pool_s pool, unsigned int idx)
{
    int rc = YAKSA_SUCCESS;
    pool_head_s *pool_head = (pool_head_s *) pool;

    pthread_mutex_lock(&pool_head->mutex);

    chunk_s *chunk = pool_head->chunks;
    while (idx >= pool_head->elems_in_chunk) {
        assert(chunk->next);
        chunk = chunk->next;
        idx -= pool_head->elems_in_chunk;
    }

    chunk->elems[idx] = NULL;

    pthread_mutex_unlock(&pool_head->mutex);
    return rc;
}

int yaksu_pool_elem_get(yaksu_pool_s pool, unsigned int elem_idx, void **elem)
{
    int rc = YAKSA_SUCCESS;
    pool_head_s *pool_head = (pool_head_s *) pool;
    unsigned int idx = elem_idx;

    chunk_s *chunk = pool_head->chunks;
    while (idx >= pool_head->elems_in_chunk) {
        assert(chunk->next);
        chunk = chunk->next;
        idx -= pool_head->elems_in_chunk;
    }

    *elem = chunk->elems[idx];

    return rc;
}
