/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_CUDAI_H_INCLUDED
#define YAKSURI_CUDAI_H_INCLUDED

#include "yaksi.h"
#include <cuda_runtime_api.h>

#define YAKSURI_CUDAI_THREAD_BLOCK_SIZE  (256)
#define YAKSURI_CUDAI_PUP_BUF_SIZE   (64 * 1024)

/* *INDENT-OFF* */
#ifdef __cplusplus
extern "C" {
#endif
/* *INDENT-ON* */

#define YAKSURI_CUDAI_CUDA_ERR_CHECK(cerr)                              \
    do {                                                                \
        if (cerr != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA Error (%s:%s,%d): %s\n", __func__, __FILE__, __LINE__, cudaGetErrorString(cerr)); \
        }                                                               \
    } while (0)

#define YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail)            \
    do {                                                                \
        if (cerr != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA Error (%s:%s,%d): %s\n", __func__, __FILE__, __LINE__, cudaGetErrorString(cerr)); \
            rc = YAKSA_ERR__INTERNAL;                                   \
            goto fn_fail;                                               \
        }                                                               \
    } while (0)

typedef struct {
    yaksu_pool_s pup_buf_pool;
    cudaStream_t stream;
} yaksuri_cudai_global_s;
extern yaksuri_cudai_global_s yaksuri_cudai_global;

typedef struct yaksuri_cudai_md_s {
    union {
        struct {
            int count;
            intptr_t stride;
            struct yaksuri_cudai_md_s *child;
        } contig;
        struct {
            struct yaksuri_cudai_md_s *child;
        } dup;
        struct {
            struct yaksuri_cudai_md_s *child;
        } resized;
        struct {
            int count;
            int blocklength;
            intptr_t stride;
            struct yaksuri_cudai_md_s *child;
        } hvector;
        struct {
            int count;
            int blocklength;
            intptr_t *array_of_displs;
            struct yaksuri_cudai_md_s *child;
        } blkhindx;
        struct {
            int count;
            int *array_of_blocklengths;
            intptr_t *array_of_displs;
            struct yaksuri_cudai_md_s *child;
        } hindexed;
    } u;

    uintptr_t extent;
    uintptr_t num_elements;
} yaksuri_cudai_md_s;


int yaksuri_cudai_md_alloc(yaksi_type_s * type);
int yaksuri_cudai_populate_pupfns(yaksi_type_s * type);

/* *INDENT-OFF* */
#ifdef __cplusplus
}
#endif
/* *INDENT-ON* */

#endif /* YAKSURI_CUDAI_H_INCLUDED */
