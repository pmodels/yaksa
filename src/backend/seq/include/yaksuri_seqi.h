/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_SEQI_H_INCLUDED
#define YAKSURI_SEQI_H_INCLUDED

#include "yaksi.h"

typedef struct yaksuri_seqi_type_s {
    int (*pack) (const void *inbuf, void *outbuf, uintptr_t count, struct yaksi_type_s *);
    int (*unpack) (const void *inbuf, void *outbuf, uintptr_t count, struct yaksi_type_s *);
} yaksuri_seqi_type_s;

int yaksuri_seqi_populate_pupfns(yaksi_type_s * type);

#endif /* YAKSURI_SEQI_H_INCLUDED */
