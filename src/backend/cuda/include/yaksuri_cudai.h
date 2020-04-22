/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_CUDAI_H_INCLUDED
#define YAKSURI_CUDAI_H_INCLUDED

#include "yaksi.h"
#include <stdint.h>
#include <pthread.h>
#include <cuda_runtime_api.h>

#define CUDA_P2P_ENABLED  (1)
#define CUDA_P2P_DISABLED (2)
#define CUDA_P2P_CLIQUES  (3)

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
    int ndevices;
    cudaStream_t *stream;
    bool **p2p;
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

typedef struct yaksuri_cudai_type_s {
    void (*pack) (const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s * md,
                  int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device);
    void (*unpack) (const void *inbuf, void *outbuf, uintptr_t count, yaksuri_cudai_md_s * md,
                    int n_threads, int n_blocks_x, int n_blocks_y, int n_blocks_z, int device);
    yaksuri_cudai_md_s *md;
    pthread_mutex_t mdmutex;
    uintptr_t num_elements;
} yaksuri_cudai_type_s;

int yaksuri_cudai_finalize_hook(void);
int yaksuri_cudai_type_create_hook(yaksi_type_s * type);
int yaksuri_cudai_type_free_hook(yaksi_type_s * type);

int yaksuri_cudai_event_destroy(void *event);
int yaksuri_cudai_event_query(void *event, int *completed);
int yaksuri_cudai_event_synchronize(void *event);

int yaksuri_cudai_get_ptr_attr(const void *buf, yaksur_ptr_attr_s * ptrattr);

int yaksuri_cudai_md_alloc(yaksi_type_s * type);
int yaksuri_cudai_populate_pupfns(yaksi_type_s * type);

int yaksuri_cudai_ipack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                        void *gpu_tmpbuf, void *interm_event, void **event);
int yaksuri_cudai_iunpack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                          void *gpu_tmpbuf, void *interm_event, void **event);
int yaksuri_cudai_pup_is_supported(yaksi_type_s * type, bool * is_supported);

/* *INDENT-OFF* */
#ifdef __cplusplus
}
#endif
/* *INDENT-ON* */

#endif /* YAKSURI_CUDAI_H_INCLUDED */
