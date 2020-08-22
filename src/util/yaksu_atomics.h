/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSU_ATOMICS_H_INCLUDED
#define YAKSU_ATOMICS_H_INCLUDED

#include <yaksa_config.h>

#ifdef HAVE_STDATOMIC_H

#include <stdatomic.h>

typedef atomic_int yaksu_atomic_int;

static inline int yaksu_atomic_incr(yaksu_atomic_int * val)
{
    return atomic_fetch_add(val, 1);
}

static inline int yaksu_atomic_decr(yaksu_atomic_int * val)
{
    return atomic_fetch_sub(val, 1);
}

static inline int yaksu_atomic_load(yaksu_atomic_int * val)
{
    return atomic_load_explicit(val, memory_order_acquire);
}

static inline void yaksu_atomic_store(yaksu_atomic_int * val, int x)
{
    return atomic_store_explicit(val, x, memory_order_release);
}

#else

#include <pthread.h>
#include "yaksu_rwlocks.h"

extern yaksu_rwlock_t yaksui_atomic_lock;
typedef int yaksu_atomic_int;

static inline int yaksu_atomic_incr(yaksu_atomic_int * val)
{
    yaksu_rwlock_wrlock(&yaksui_atomic_lock);
    int ret = (*val)++;
    yaksu_rwlock_unlock(&yaksui_atomic_lock);

    return ret;
}

static inline int yaksu_atomic_decr(yaksu_atomic_int * val)
{
    yaksu_rwlock_wrlock(&yaksui_atomic_lock);
    int ret = (*val)--;
    yaksu_rwlock_unlock(&yaksui_atomic_lock);

    return ret;
}

static inline int yaksu_atomic_load(yaksu_atomic_int * val)
{
    yaksu_rwlock_rdlock(&yaksui_atomic_lock);
    int ret = (*val);
    yaksu_rwlock_unlock(&yaksui_atomic_lock);

    return ret;
}

static inline void yaksu_atomic_store(yaksu_atomic_int * val, int x)
{
    yaksu_rwlock_wrlock(&yaksui_atomic_lock);
    *val = x;
    yaksu_rwlock_unlock(&yaksui_atomic_lock);
}

#endif

#endif /* YAKSU_THREADS_H_INCLUDED */
