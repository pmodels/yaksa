/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_H_INCLUDED
#define YAKSURI_H_INCLUDED

#include "yaksi.h"

typedef enum yaksuri_gpudriver_id_e {
    YAKSURI_GPUDRIVER_ID__UNSET = -1,
    YAKSURI_GPUDRIVER_ID__CUDA = 0,
    YAKSURI_GPUDRIVER_ID__LAST,
} yaksuri_gpudriver_id_e;

typedef enum yaksuri_pup_e {
    YAKSURI_PUPTYPE__PACK,
    YAKSURI_PUPTYPE__UNPACK,
} yaksuri_puptype_e;

typedef struct {
    void *slab;
    uintptr_t slab_head_offset;
    uintptr_t slab_tail_offset;
} yaksuri_slab_s;

typedef struct {
    struct {
        yaksuri_slab_s host;
        yaksuri_slab_s *device;
        yaksur_gpudriver_info_s *info;
    } gpudriver[YAKSURI_GPUDRIVER_ID__LAST];
} yaksuri_global_s;
extern yaksuri_global_s yaksuri_global;

typedef struct {
    yaksuri_gpudriver_id_e gpudriver_id;
    void *event;

    enum {
        YAKSURI_REQUEST_KIND__UNSET,
        YAKSURI_REQUEST_KIND__DIRECT,
        YAKSURI_REQUEST_KIND__STAGED,
    } kind;
} yaksuri_request_s;

int yaksuri_progress_enqueue(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                             yaksi_info_s * info, yaksi_request_s * request,
                             yaksur_ptr_attr_s inattr, yaksur_ptr_attr_s outattr,
                             yaksuri_puptype_e puptype);
int yaksuri_progress_poke(void);

#endif /* YAKSURI_H_INCLUDED */
