/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSU_MEM_H_INCLUDED
#define YAKSU_MEM_H_INCLUDED

#define YAKSU_MEMCPY__SYSTEM        0

static inline void *yaksu_memcpy(void *dest, const void *src, size_t n, int memcpy_type)
{
    switch(memcpy_type) {
        case YAKSU_MEMCPY__SYSTEM:
            return memcpy(dest, src, n);
        default:
            /* Using an invalid memcpy type */
            assert(0);
    }
}

#endif /* YAKSU_MEM_H_INCLUDED */