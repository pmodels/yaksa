/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_H_INCLUDED
#define YAKSURI_H_INCLUDED

#include "yaksi.h"

typedef struct {
    struct {
        struct {
            void *slab;
            uintptr_t slab_head_offset;
            uintptr_t slab_tail_offset;

            yaksu_malloc_fn malloc;
            yaksu_free_fn free;
        } device, host;
    } cuda;
} yaksuri_global_s;
extern yaksuri_global_s yaksuri_global;

typedef enum {
    YAKSURI_PROGRESS_ELEM_KIND__PACK_D2RH,
    YAKSURI_PROGRESS_ELEM_KIND__PACK_D2URH,
    YAKSURI_PROGRESS_ELEM_KIND__PACK_RH2D,
    YAKSURI_PROGRESS_ELEM_KIND__PACK_URH2D,
    YAKSURI_PROGRESS_ELEM_KIND__UNPACK_D2RH,
    YAKSURI_PROGRESS_ELEM_KIND__UNPACK_D2URH,
    YAKSURI_PROGRESS_ELEM_KIND__UNPACK_RH2D,
    YAKSURI_PROGRESS_ELEM_KIND__UNPACK_URH2D,
} yaksuri_progress_elem_kind_e;

int yaksuri_progress_enqueue(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                             yaksi_request_s * request, yaksuri_progress_elem_kind_e kind);
int yaksuri_progress_poke(void);

#endif /* YAKSURI_H_INCLUDED */
