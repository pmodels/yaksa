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

typedef struct pool_elem {
    void **elems;
    void *slab;
    struct pool_elem *next;
} pool_elem_s;

typedef struct pool_head {
    uintptr_t elemsize;
    uintptr_t elems_in_chunk;
    uintptr_t maxelems;

    yaksu_malloc_fn malloc_fn;
    yaksu_free_fn free_fn;

    pthread_mutex_t mutex;

    uintptr_t num_pool_elems;
    uintptr_t max_pool_elems;
    pool_elem_s *pool_elems;
} pool_head_s;

static pthread_mutex_t global_mutex = PTHREAD_MUTEX_INITIALIZER;

int yaksu_pool_alloc(uintptr_t elemsize, uintptr_t elems_in_chunk, uintptr_t maxelems,
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

    pool_head->num_pool_elems = 0;
    pool_head->max_pool_elems = maxelems / elems_in_chunk;
    pool_head->pool_elems = NULL;

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
    for (pool_elem_s * tmp = pool_head->pool_elems; tmp;) {
        pool_elem_s *next = tmp->next;
        for (int i = 0; i < pool_head->elems_in_chunk; i++)
            if (tmp->elems[i])
                count++;

        pool_head->free_fn(tmp->slab);
        free(tmp);
        tmp = next;
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

int yaksu_pool_elem_alloc(yaksu_pool_s pool, void **elem, int *elem_idx)
{
    int rc = YAKSA_SUCCESS;
    pool_head_s *pool_head = (pool_head_s *) pool;
    int idx;

    pthread_mutex_lock(&pool_head->mutex);

    /* try to find an available type */
    idx = 0;
    for (pool_elem_s * tmp = pool_head->pool_elems; tmp; tmp = tmp->next) {
        for (int i = 0; i < pool_head->elems_in_chunk; i++) {
            if (tmp->elems[i] == NULL) {
                tmp->elems[i] = (char *) tmp->slab + i * pool_head->elemsize;
                YAKSU_ERR_CHKANDJUMP(!tmp->elems[i], rc, YAKSA_ERR__OUT_OF_MEM, fn_fail);
                *elem_idx = idx;
                *elem = tmp->elems[i];
                goto fn_exit;
            }
            idx++;
        }
    }

    /* no empty slot available; see if we can allocate more chunks */
    assert(pool_head->num_pool_elems <= pool_head->max_pool_elems);
    if (pool_head->num_pool_elems == pool_head->max_pool_elems)
        goto fn_exit;

    /* allocate another chunk */
    idx = 0;
    pool_elem_s *new = (pool_elem_s *) malloc(sizeof(pool_elem_s));
    YAKSU_ERR_CHKANDJUMP(!new, rc, YAKSA_ERR__OUT_OF_MEM, fn_fail);

    new->slab = pool_head->malloc_fn(pool_head->elems_in_chunk * pool_head->elemsize);
    YAKSU_ERR_CHKANDJUMP(!new->slab, rc, YAKSA_ERR__OUT_OF_MEM, fn_fail);

    new->elems = (void **) malloc(pool_head->elems_in_chunk * sizeof(void *));
    YAKSU_ERR_CHKANDJUMP(!new->elems, rc, YAKSA_ERR__OUT_OF_MEM, fn_fail);

    memset(new->elems, 0, pool_head->elems_in_chunk * sizeof(void *));
    new->next = NULL;

    pool_head->num_pool_elems++;

    if (pool_head->pool_elems == NULL) {
        pool_head->pool_elems = new;
    } else {
        pool_elem_s *tmp;
        idx += pool_head->elems_in_chunk;
        for (tmp = pool_head->pool_elems; tmp->next; tmp = tmp->next) {
            idx += pool_head->elems_in_chunk;
        }
        tmp->next = new;
    }

    new->elems[0] = new->slab;
    *elem_idx = idx;
    *elem = new->elems[0];

  fn_exit:
    pthread_mutex_unlock(&pool_head->mutex);
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksu_pool_elem_free(yaksu_pool_s pool, int idx)
{
    int rc = YAKSA_SUCCESS;
    pool_head_s *pool_head = (pool_head_s *) pool;

    pthread_mutex_lock(&pool_head->mutex);

    pool_elem_s *tmp = pool_head->pool_elems;
    while (idx >= pool_head->elems_in_chunk) {
        assert(tmp->next);
        tmp = tmp->next;
        idx -= pool_head->elems_in_chunk;
    }

    tmp->elems[idx] = NULL;

    pthread_mutex_unlock(&pool_head->mutex);
    return rc;
}

int yaksu_pool_elem_get(yaksu_pool_s pool, int elem_idx, void **elem)
{
    int rc = YAKSA_SUCCESS;
    pool_head_s *pool_head = (pool_head_s *) pool;
    int idx = elem_idx;

    pool_elem_s *tmp = pool_head->pool_elems;
    while (idx >= pool_head->elems_in_chunk) {
        assert(tmp->next);
        tmp = tmp->next;
        idx -= pool_head->elems_in_chunk;
    }

    *elem = tmp->elems[idx];

    return rc;
}
