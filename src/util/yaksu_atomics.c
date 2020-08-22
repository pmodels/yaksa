/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <pthread.h>
#include "yaksu_atomics.h"

yaksu_rwlock_t yaksui_atomic_lock = YAKSU_RWLOCK_INITIALIZER;
