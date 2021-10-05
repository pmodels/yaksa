/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSU_MEM_H_INCLUDED
#define YAKSU_MEM_H_INCLUDED

#ifdef HAVE_MM256_STREAM_SI256
#include <immintrin.h>
#endif
#include "string.h"

#define YAKSU_MEMCPY__SYSTEM        0
#define YAKSU_MEMCPY__STREAM        1

static inline void *yaksu_stream_memcpy(void *dest, const void *src, size_t n)
{
#ifdef HAVE_MM256_STREAM_SI256
    if (n <= 256) {
        return memcpy(dest, src, n);
    }

    char *d = (char *) dest;
    const char *s = (const char *) src;

    /* Copy the first 63 bits or less (if the address isn't 64-bit aligned) using a regular memcpy to
     * make the rest faster */
    if (((uintptr_t)d) & 63) {
        const uintptr_t t = 64 - (((uintptr_t)d) & 63);
        memcpy(d, s, t);
        d += t;
        s += t;
        n -= t;
    }

    /* Copy 256, 128, and then 64 bit chunks using the non-temporal store to pipeline as much as
     * possible. */
    while (n >= 256) {
        __m256i ymm0 = _mm256_loadu_si256((__m256i const *)(s + (32 * 0)));
        __m256i ymm1 = _mm256_loadu_si256((__m256i const *)(s + (32 * 1)));
        __m256i ymm2 = _mm256_loadu_si256((__m256i const *)(s + (32 * 2)));
        __m256i ymm3 = _mm256_loadu_si256((__m256i const *)(s + (32 * 3)));
        __m256i ymm4 = _mm256_loadu_si256((__m256i const *)(s + (32 * 4)));
        __m256i ymm5 = _mm256_loadu_si256((__m256i const *)(s + (32 * 5)));
        __m256i ymm6 = _mm256_loadu_si256((__m256i const *)(s + (32 * 6)));
        __m256i ymm7 = _mm256_loadu_si256((__m256i const *)(s + (32 * 7)));
        _mm256_stream_si256((__m256i *)(d + (32 * 0)), ymm0);
        _mm256_stream_si256((__m256i *)(d + (32 * 1)), ymm1);
        _mm256_stream_si256((__m256i *)(d + (32 * 2)), ymm2);
        _mm256_stream_si256((__m256i *)(d + (32 * 3)), ymm3);
        _mm256_stream_si256((__m256i *)(d + (32 * 4)), ymm4);
        _mm256_stream_si256((__m256i *)(d + (32 * 5)), ymm5);
        _mm256_stream_si256((__m256i *)(d + (32 * 6)), ymm6);
        _mm256_stream_si256((__m256i *)(d + (32 * 7)), ymm7);
        d += 256;
        s += 256;
        n -= 256;
    }

    if (n & 128) {
        __m256i ymm0 = _mm256_loadu_si256((__m256i const *)(s + (32 * 0)));
        __m256i ymm1 = _mm256_loadu_si256((__m256i const *)(s + (32 * 1)));
        __m256i ymm2 = _mm256_loadu_si256((__m256i const *)(s + (32 * 2)));
        __m256i ymm3 = _mm256_loadu_si256((__m256i const *)(s + (32 * 3)));
        _mm256_stream_si256((__m256i *)(d + (32 * 0)), ymm0);
        _mm256_stream_si256((__m256i *)(d + (32 * 1)), ymm1);
        _mm256_stream_si256((__m256i *)(d + (32 * 2)), ymm2);
        _mm256_stream_si256((__m256i *)(d + (32 * 3)), ymm3);
        d += 128;
        s += 128;
    }

    if (n & 64) {
        __m256i ymm0 = _mm256_loadu_si256((__m256i const *)(s + (32 * 0)));
        __m256i ymm1 = _mm256_loadu_si256((__m256i const *)(s + (32 * 1)));
        _mm256_stream_si256((__m256i *)(d + (32 * 0)), ymm0);
        _mm256_stream_si256((__m256i *)(d + (32 * 1)), ymm1);
        d += 64;
        s += 64;
    }

    /* If there is any data left, copy it using a regular memcpy */
    if (n & 63) {
        memcpy(d, s, (n & 63));
    }

    _mm_sfence();

    return d;
#else
    return memcpy(dest, src, n);
#endif
}

static inline void *yaksu_memcpy(void *dest, const void *src, size_t n, int memcpy_type)
{
    switch(memcpy_type) {
        case YAKSU_MEMCPY__STREAM:
            return yaksu_stream_memcpy(dest, src, n);
        case YAKSU_MEMCPY__SYSTEM:
            return memcpy(dest, src, n);
        default:
            /* Using an invalid memcpy type so fall back to the default memcpy implementation. */
    return memcpy(dest, src, n);
    }
}

#endif /* YAKSU_MEM_H_INCLUDED */
