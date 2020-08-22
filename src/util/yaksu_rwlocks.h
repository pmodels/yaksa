/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSU_RWLOCK_H_INCLUDED
#define YAKSU_RWLOCK_H_INCLUDED

#include <yaksa_config.h>

#if defined(HAVE_PTHREAD_RWLOCKS)

#define YAKSU_RWLOCK_INITIALIZER PTHREAD_RWLOCK_INITIALIZER
#define yaksu_rwlock_t pthread_rwlock_t
#define yaksu_rwlock_wrlock pthread_rwlock_wrlock
#define yaksu_rwlock_rdlock pthread_rwlock_rdlock
#define yaksu_rwlock_unlock pthread_rwlock_unlock
#define yaksu_rwlock_init pthread_rwlock_init
#define yaksu_rwlock_destroy pthread_rwlock_destroy

#else

#define YAKSU_RWLOCK_INITIALIZER PTHREAD_MUTEX_INITIALIZER
#define yaksu_rwlock_t pthread_mutex_t
#define yaksu_rwlock_wrlock pthread_mutex_lock
#define yaksu_rwlock_rdlock pthread_mutex_lock
#define yaksu_rwlock_unlock pthread_mutex_unlock
#define yaksu_rwlock_init pthread_mutex_init
#define yaksu_rwlock_destroy pthread_mutex_destroy

#endif /* HAVE_PTHREAD_RWLOCK_T */

#endif /* YAKSU_RWLOCK_H_INCLUDED */
