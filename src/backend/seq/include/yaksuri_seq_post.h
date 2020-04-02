/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSURI_SEQ_POST_H_INCLUDED
#define YAKSURI_SEQ_POST_H_INCLUDED

int yaksuri_seq_init_hook(void);
int yaksuri_seq_finalize_hook(void);
int yaksuri_seq_type_create_hook(yaksi_type_s * type);
int yaksuri_seq_type_free_hook(yaksi_type_s * type);
int yaksuri_seq_request_create_hook(yaksi_request_s * request);
int yaksuri_seq_request_free_hook(yaksi_request_s * request);
int yaksuri_seq_request_test(yaksi_request_s * request);
int yaksuri_seq_request_wait(yaksi_request_s * request);

#endif /* YAKSURI_SEQ_H_INCLUDED */
