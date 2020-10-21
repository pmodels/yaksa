/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <pthread.h>
#include "yaksa.h"
#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri.h"
#include "yutlist.h"

static yaksuri_request_s *pending_reqs = NULL;
static pthread_mutex_t progress_mutex = PTHREAD_MUTEX_INITIALIZER;

static int icopy(yaksuri_gpudriver_id_e id, const void *inbuf, void *outbuf, uintptr_t bytes,
                 struct yaksi_info_s *info, int device)
{
    int rc = YAKSA_SUCCESS;

    yaksi_type_s *byte_type;
    rc = yaksi_type_get(YAKSA_TYPE__BYTE, &byte_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksuri_global.gpudriver[id].info->ipack(inbuf, outbuf, bytes, byte_type, info, device);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int ipack(yaksuri_gpudriver_id_e id, const void *inbuf, void *outbuf, uintptr_t count,
                 struct yaksi_type_s *type, struct yaksi_info_s *info, int device)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].info->ipack(inbuf, outbuf, count, type, info, device);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int iunpack(yaksuri_gpudriver_id_e id, const void *inbuf, void *outbuf, uintptr_t count,
                   struct yaksi_type_s *type, struct yaksi_info_s *info, int device)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].info->iunpack(inbuf, outbuf, count, type, info, device);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int check_p2p_comm(yaksuri_gpudriver_id_e id, int indev, int outdev, bool * is_enabled)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].info->check_p2p_comm(indev, outdev, is_enabled);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int event_create(yaksuri_gpudriver_id_e id, int device, void **event)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].info->event_create(device, event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int event_destroy(yaksuri_gpudriver_id_e id, void *event)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].info->event_destroy(event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int event_record(yaksuri_gpudriver_id_e id, void *event)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].info->event_record(event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int event_query(yaksuri_gpudriver_id_e id, void *event, int *completed)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].info->event_query(event, completed);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int event_add_dependency(yaksuri_gpudriver_id_e id, void *event, int device)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].info->event_add_dependency(event, device);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int alloc_chunk(yaksuri_request_s * backend, yaksuri_subreq_s * subreq,
                       yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;

    assert(subreq);
    assert(subreq->kind == YAKSURI_SUBREQ_KIND__MULTI_CHUNK);

    /* allocate the chunk */
    *chunk = (yaksuri_subreq_chunk_s *) malloc(sizeof(yaksuri_subreq_chunk_s));

    (*chunk)->count_offset = subreq->u.multiple.issued_count;
    uintptr_t count_per_chunk = YAKSURI_TMPBUF_EL_SIZE / subreq->u.multiple.type->size;
    if ((*chunk)->count_offset + count_per_chunk <= subreq->u.multiple.count) {
        (*chunk)->count = count_per_chunk;
    } else {
        (*chunk)->count = subreq->u.multiple.count - (*chunk)->count_offset;
    }

    (*chunk)->event = NULL;

    DL_APPEND(subreq->u.multiple.chunks, (*chunk));

    return rc;
}

static int simple_release(yaksuri_request_s * backend, yaksuri_subreq_s * subreq,
                          yaksuri_subreq_chunk_s * chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = backend->gpudriver_id;

    /* cleanup */
    if (chunk->event) {
        rc = event_destroy(id, chunk->event);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    for (int i = 0; i < chunk->num_tmpbufs; i++) {
        rc = yaksu_buffer_pool_elem_free(chunk->tmpbufs[i].pool, chunk->tmpbufs[i].buf);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    DL_DELETE(subreq->u.multiple.chunks, chunk);
    free(chunk);

    if (subreq->u.multiple.chunks == NULL) {
        DL_DELETE(backend->subreqs, subreq);
        free(subreq);
    }
    if (backend->subreqs == NULL) {
        HASH_DEL(pending_reqs, backend);
        yaksu_atomic_decr(&backend->request->cc);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_d2d_acquire(yaksuri_request_s * backend, yaksuri_subreq_s * subreq,
                            yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = backend->gpudriver_id;

    assert(backend->inattr.device != backend->outattr.device);

    *chunk = NULL;

    bool is_enabled;
    rc = check_p2p_comm(id, backend->inattr.device, backend->outattr.device, &is_enabled);
    YAKSU_ERR_CHECK(rc, fn_fail);
    assert(is_enabled);

    if (is_enabled) {
        /* p2p is enabled: we need a temporary buffer on the source device */
        void *d_buf;
        rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.
                                          gpudriver[id].device[backend->inattr.device], &d_buf);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (d_buf == NULL)
            goto fn_exit;

        /* we have the temporary buffer, so we can safely issue this
         * operation */
        rc = alloc_chunk(backend, subreq, chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);

        (*chunk)->num_tmpbufs = 1;
        (*chunk)->tmpbufs[0].buf = d_buf;
        (*chunk)->tmpbufs[0].pool = yaksuri_global.gpudriver[id].device[backend->inattr.device];

        /* first pack data from the origin buffer into the temporary buffer */
        const char *sbuf =
            (const char *) subreq->u.multiple.inbuf +
            (*chunk)->count_offset * subreq->u.multiple.type->extent;

        rc = ipack(id, sbuf, d_buf, (*chunk)->count, subreq->u.multiple.type, backend->info,
                   backend->inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        /* second copy the data into the target device */
        char *dbuf =
            (char *) subreq->u.multiple.outbuf +
            (*chunk)->count_offset * subreq->u.multiple.type->size;

        rc = icopy(id, d_buf, dbuf, (*chunk)->count * subreq->u.multiple.type->size, backend->info,
                   backend->inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        event_create(id, backend->inattr.device, &(*chunk)->event);
        event_record(id, (*chunk)->event);
    } else {
        /* p2p is not enabled: we need two temporary buffers, one on
         * the source device and one on the host */
        void *d_buf, *rh_buf;

        rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.
                                          gpudriver[id].device[backend->inattr.device], &d_buf);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (d_buf == NULL)
            goto fn_exit;

        rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].host, &rh_buf);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (rh_buf == NULL) {
            if (d_buf) {
                rc = yaksu_buffer_pool_elem_free(yaksuri_global.
                                                 gpudriver[id].device[backend->inattr.device],
                                                 d_buf);
                YAKSU_ERR_CHECK(rc, fn_fail);
            }
            goto fn_exit;
        }

        /* we have the temporary buffers, so we can safely issue this
         * operation */
        rc = alloc_chunk(backend, subreq, chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);

        (*chunk)->num_tmpbufs = 2;
        (*chunk)->tmpbufs[0].buf = d_buf;
        (*chunk)->tmpbufs[0].pool = yaksuri_global.gpudriver[id].device[backend->inattr.device];
        (*chunk)->tmpbufs[1].buf = rh_buf;
        (*chunk)->tmpbufs[1].pool = yaksuri_global.gpudriver[id].host;

        /* first pack data from the origin buffer into the temporary buffer */
        const char *sbuf =
            (const char *) subreq->u.multiple.inbuf +
            (*chunk)->count_offset * subreq->u.multiple.type->extent;

        rc = ipack(id, sbuf, d_buf, (*chunk)->count, subreq->u.multiple.type, backend->info,
                   backend->inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        /* second copy the data into the temporary host buffer */
        rc = icopy(id, d_buf, rh_buf, (*chunk)->count * subreq->u.multiple.type->size,
                   backend->info, backend->inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        /* third DMA from the host temporary buffer to the target device */
        void *event;
        rc = event_create(id, backend->inattr.device, &event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_record(id, event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_add_dependency(id, event, backend->outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_destroy(id, event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        char *dbuf =
            (char *) subreq->u.multiple.outbuf +
            (*chunk)->count_offset * subreq->u.multiple.type->size;

        rc = icopy(id, rh_buf, dbuf, (*chunk)->count * subreq->u.multiple.type->size,
                   backend->info, backend->outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_create(id, backend->outattr.device, &(*chunk)->event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_record(id, (*chunk)->event);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_d2rh_acquire(yaksuri_request_s * backend, yaksuri_subreq_s * subreq,
                             yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = backend->gpudriver_id;

    *chunk = NULL;

    /* we need a temporary buffer on the source device */
    void *d_buf;
    rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].device[backend->inattr.device],
                                      &d_buf);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (d_buf == NULL)
        goto fn_exit;

    /* we have the temporary buffer, so we can safely issue this
     * operation */
    rc = alloc_chunk(backend, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    (*chunk)->num_tmpbufs = 1;
    (*chunk)->tmpbufs[0].buf = d_buf;
    (*chunk)->tmpbufs[0].pool = yaksuri_global.gpudriver[id].device[backend->inattr.device];

    /* first pack data from the origin buffer into the temporary buffer */
    const char *sbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;

    rc = ipack(id, sbuf, d_buf, (*chunk)->count, subreq->u.multiple.type, backend->info,
               backend->inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* second copy the data into the destination buffer */
    char *dbuf;
    dbuf =
        (char *) subreq->u.multiple.outbuf + (*chunk)->count_offset * subreq->u.multiple.type->size;

    rc = icopy(id, d_buf, dbuf, (*chunk)->count * subreq->u.multiple.type->size, backend->info,
               backend->inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    event_create(id, backend->inattr.device, &(*chunk)->event);
    event_record(id, (*chunk)->event);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_d2urh_acquire(yaksuri_request_s * backend, yaksuri_subreq_s * subreq,
                              yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = backend->gpudriver_id;

    *chunk = NULL;

    /* we need two temporary buffers, one on the source device and one
     * on the host */
    void *d_buf, *rh_buf;

    rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].device[backend->inattr.device],
                                      &d_buf);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (d_buf == NULL)
        goto fn_exit;

    rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].host, &rh_buf);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (rh_buf == NULL) {
        if (d_buf) {
            rc = yaksu_buffer_pool_elem_free(yaksuri_global.
                                             gpudriver[id].device[backend->inattr.device], d_buf);
            YAKSU_ERR_CHECK(rc, fn_fail);
        }
        goto fn_exit;
    }

    /* we have the temporary buffers, so we can safely issue this
     * operation */
    rc = alloc_chunk(backend, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    (*chunk)->num_tmpbufs = 2;
    (*chunk)->tmpbufs[0].buf = d_buf;
    (*chunk)->tmpbufs[0].pool = yaksuri_global.gpudriver[id].device[backend->inattr.device];
    (*chunk)->tmpbufs[1].buf = rh_buf;
    (*chunk)->tmpbufs[1].pool = yaksuri_global.gpudriver[id].host;

    /* first pack data from the origin buffer into the temporary buffer */
    const char *sbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;

    rc = ipack(id, sbuf, d_buf, (*chunk)->count, subreq->u.multiple.type, backend->info,
               backend->inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* second copy the data into the temporary host buffer */
    rc = icopy(id, d_buf, rh_buf, (*chunk)->count * subreq->u.multiple.type->size,
               backend->info, backend->inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = event_create(id, backend->inattr.device, &(*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = event_record(id, (*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_d2urh_release(yaksuri_request_s * backend, yaksuri_subreq_s * subreq,
                              yaksuri_subreq_chunk_s * chunk)
{
    int rc = YAKSA_SUCCESS;

    yaksi_type_s *byte_type;
    rc = yaksi_type_get(YAKSA_TYPE__BYTE, &byte_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    char *dbuf;
    dbuf = (char *) subreq->u.multiple.outbuf + chunk->count_offset * subreq->u.multiple.type->size;
    rc = yaksuri_seq_ipack(chunk->tmpbufs[1].buf, dbuf,
                           chunk->count * subreq->u.multiple.type->size, byte_type, backend->info);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = simple_release(backend, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_h2d_acquire(yaksuri_request_s * backend, yaksuri_subreq_s * subreq,
                            yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = backend->gpudriver_id;

    *chunk = NULL;

    /* we need a host temporary buffer */
    void *rh_buf;

    rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].host, &rh_buf);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (rh_buf == NULL)
        goto fn_exit;

    /* we have the temporary buffers, so we can safely issue this
     * operation */
    rc = alloc_chunk(backend, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    (*chunk)->num_tmpbufs = 1;
    (*chunk)->tmpbufs[0].buf = rh_buf;
    (*chunk)->tmpbufs[0].pool = yaksuri_global.gpudriver[id].host;

    /* first pack data from the origin buffer into the temporary buffer */
    const char *sbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;

    rc = yaksuri_seq_ipack(sbuf, rh_buf, (*chunk)->count, subreq->u.multiple.type, backend->info);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* second copy the data into the target device */
    char *dbuf;
    dbuf =
        (char *) subreq->u.multiple.outbuf + (*chunk)->count_offset * subreq->u.multiple.type->size;

    rc = icopy(id, rh_buf, dbuf, (*chunk)->count * subreq->u.multiple.type->size, backend->info,
               backend->outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    event_create(id, backend->outattr.device, &(*chunk)->event);
    event_record(id, (*chunk)->event);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int unpack_d2d_acquire(yaksuri_request_s * backend, yaksuri_subreq_s * subreq,
                              yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = backend->gpudriver_id;

    assert(backend->inattr.device != backend->outattr.device);

    *chunk = NULL;

    bool is_enabled;
    rc = check_p2p_comm(id, backend->inattr.device, backend->outattr.device, &is_enabled);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (is_enabled) {
        /* p2p is enabled: we need a temporary buffer on the destination device */
        void *d_buf;
        rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.
                                          gpudriver[id].device[backend->outattr.device], &d_buf);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (d_buf == NULL)
            goto fn_exit;

        /* we have the temporary buffer, so we can safely issue this
         * operation */
        rc = alloc_chunk(backend, subreq, chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);

        (*chunk)->num_tmpbufs = 1;
        (*chunk)->tmpbufs[0].buf = d_buf;
        (*chunk)->tmpbufs[0].pool = yaksuri_global.gpudriver[id].device[backend->outattr.device];

        /* first copy the data from the origin buffer into the
         * temporary buffer */
        const char *sbuf;
        sbuf = (const char *) subreq->u.multiple.inbuf +
            (*chunk)->count_offset * subreq->u.multiple.type->size;

        rc = icopy(id, sbuf, d_buf, (*chunk)->count * subreq->u.multiple.type->size, backend->info,
                   backend->inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        /* second unpack the data into the destination buffer */
        void *event;
        rc = event_create(id, backend->inattr.device, &event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_record(id, event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_add_dependency(id, event, backend->outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_destroy(id, event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        char *dbuf;
        dbuf = (char *) subreq->u.multiple.outbuf +
            (*chunk)->count_offset * subreq->u.multiple.type->extent;

        rc = iunpack(id, d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type, backend->info,
                     backend->outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        event_create(id, backend->outattr.device, &(*chunk)->event);
        event_record(id, (*chunk)->event);
    } else {
        /* p2p is not enabled: we need two temporary buffers, one on
         * the destination device and one on the host */
        void *d_buf, *rh_buf;

        rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.
                                          gpudriver[id].device[backend->outattr.device], &d_buf);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (d_buf == NULL)
            goto fn_exit;

        rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].host, &rh_buf);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (rh_buf == NULL) {
            if (d_buf) {
                rc = yaksu_buffer_pool_elem_free(yaksuri_global.
                                                 gpudriver[id].device[backend->outattr.device],
                                                 d_buf);
                YAKSU_ERR_CHECK(rc, fn_fail);
            }
            goto fn_exit;
        }

        /* we have the temporary buffers, so we can safely issue this
         * operation */
        rc = alloc_chunk(backend, subreq, chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);

        (*chunk)->num_tmpbufs = 2;
        (*chunk)->tmpbufs[0].buf = d_buf;
        (*chunk)->tmpbufs[0].pool = yaksuri_global.gpudriver[id].device[backend->outattr.device];
        (*chunk)->tmpbufs[1].buf = rh_buf;
        (*chunk)->tmpbufs[1].pool = yaksuri_global.gpudriver[id].host;

        /* first copy data from the origin buffer into the temporary host buffer */
        const char *sbuf;
        sbuf = (const char *) subreq->u.multiple.inbuf +
            (*chunk)->count_offset * subreq->u.multiple.type->size;

        rc = icopy(id, sbuf, rh_buf, (*chunk)->count * subreq->u.multiple.type->size, backend->info,
                   backend->inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        /* second copy the data from the temporary host buffer into the
         * temporary destination device buffer */
        void *event;
        rc = event_create(id, backend->inattr.device, &event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_record(id, event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_add_dependency(id, event, backend->outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_destroy(id, event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = icopy(id, rh_buf, d_buf, (*chunk)->count * subreq->u.multiple.type->size,
                   backend->info, backend->outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        /* third unpack from the temporary device buffer to the destination buffer */
        char *dbuf;
        dbuf = (char *) subreq->u.multiple.outbuf +
            (*chunk)->count_offset * subreq->u.multiple.type->extent;

        rc = iunpack(id, d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type,
                     backend->info, backend->outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_create(id, backend->outattr.device, &(*chunk)->event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_record(id, (*chunk)->event);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int unpack_rh2d_acquire(yaksuri_request_s * backend, yaksuri_subreq_s * subreq,
                               yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = backend->gpudriver_id;

    *chunk = NULL;

    /* we need a temporary buffer on the destination device */
    void *d_buf;
    rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].device[backend->outattr.device],
                                      &d_buf);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (d_buf == NULL)
        goto fn_exit;

    /* we have the temporary buffer, so we can safely issue this
     * operation */
    rc = alloc_chunk(backend, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    (*chunk)->num_tmpbufs = 1;
    (*chunk)->tmpbufs[0].buf = d_buf;
    (*chunk)->tmpbufs[0].pool = yaksuri_global.gpudriver[id].device[backend->outattr.device];

    /* first copy the data from the origin buffer into the temporary
     * device buffer */
    const char *sbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->size;

    rc = icopy(id, sbuf, d_buf, (*chunk)->count * subreq->u.multiple.type->size, backend->info,
               backend->outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* second unpack the data into the destination buffer */
    char *dbuf;
    dbuf = (char *) subreq->u.multiple.outbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;

    rc = iunpack(id, d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type, backend->info,
                 backend->outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    event_create(id, backend->outattr.device, &(*chunk)->event);
    event_record(id, (*chunk)->event);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int unpack_urh2d_acquire(yaksuri_request_s * backend, yaksuri_subreq_s * subreq,
                                yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = backend->gpudriver_id;

    *chunk = NULL;

    /* we need two temporary buffers, one on the destination device
     * and one on the host */
    void *d_buf, *rh_buf;

    rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].device[backend->outattr.device],
                                      &d_buf);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (d_buf == NULL)
        goto fn_exit;

    rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].host, &rh_buf);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (rh_buf == NULL) {
        if (d_buf) {
            rc = yaksu_buffer_pool_elem_free(yaksuri_global.
                                             gpudriver[id].device[backend->outattr.device], d_buf);
            YAKSU_ERR_CHECK(rc, fn_fail);
        }
        goto fn_exit;
    }

    /* we have the temporary buffer, so we can safely issue this
     * operation */
    rc = alloc_chunk(backend, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    (*chunk)->num_tmpbufs = 2;
    (*chunk)->tmpbufs[0].buf = d_buf;
    (*chunk)->tmpbufs[0].pool = yaksuri_global.gpudriver[id].device[backend->outattr.device];
    (*chunk)->tmpbufs[1].buf = rh_buf;
    (*chunk)->tmpbufs[1].pool = yaksuri_global.gpudriver[id].host;

    /* first copy the data into a temporary host buffer */
    const char *sbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->size;

    yaksi_type_s *byte_type;
    rc = yaksi_type_get(YAKSA_TYPE__BYTE, &byte_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksuri_seq_ipack(sbuf, rh_buf, (*chunk)->count * subreq->u.multiple.type->size,
                           byte_type, backend->info);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* second copy the data from the origin buffer into the temporary
     * buffer */
    rc = icopy(id, rh_buf, d_buf, (*chunk)->count * subreq->u.multiple.type->size, backend->info,
               backend->outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* third unpack the data into the destination buffer */
    char *dbuf;
    dbuf = (char *) subreq->u.multiple.outbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;

    rc = iunpack(id, d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type, backend->info,
                 backend->outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    event_create(id, backend->outattr.device, &(*chunk)->event);
    event_record(id, (*chunk)->event);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int unpack_d2h_acquire(yaksuri_request_s * backend, yaksuri_subreq_s * subreq,
                              yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = backend->gpudriver_id;

    *chunk = NULL;

    /* we need a temporary buffer on the host */
    void *rh_buf;
    rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].host, &rh_buf);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (rh_buf == NULL)
        goto fn_exit;

    /* we have the temporary buffer, so we can safely issue this
     * operation */
    rc = alloc_chunk(backend, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    (*chunk)->num_tmpbufs = 1;
    (*chunk)->tmpbufs[0].buf = rh_buf;
    (*chunk)->tmpbufs[0].pool = yaksuri_global.gpudriver[id].host;

    /* first copy the data from the origin buffer into the temporary
     * host buffer */
    const char *sbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->size;

    rc = icopy(id, sbuf, rh_buf, (*chunk)->count * subreq->u.multiple.type->size, backend->info,
               backend->inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    event_create(id, backend->inattr.device, &(*chunk)->event);
    event_record(id, (*chunk)->event);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int unpack_d2h_release(yaksuri_request_s * backend, yaksuri_subreq_s * subreq,
                              yaksuri_subreq_chunk_s * chunk)
{
    int rc = YAKSA_SUCCESS;

    char *dbuf;
    dbuf =
        (char *) subreq->u.multiple.outbuf + chunk->count_offset * subreq->u.multiple.type->extent;
    rc = yaksuri_seq_iunpack(chunk->tmpbufs[0].buf, dbuf, chunk->count, subreq->u.multiple.type,
                             backend->info);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = simple_release(backend, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_progress_enqueue(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                             yaksi_info_s * info, yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_request_s *backend = (yaksuri_request_s *) request->backend.priv;
    yaksuri_gpudriver_id_e id = backend->gpudriver_id;

    assert(yaksuri_global.gpudriver[id].info);
    assert(backend->inattr.type == YAKSUR_PTR_TYPE__GPU ||
           backend->outattr.type == YAKSUR_PTR_TYPE__GPU);

    backend->info = info;

    /* if the GPU backend cannot support this type, return */
    bool is_supported;
    rc = yaksuri_global.gpudriver[id].info->pup_is_supported(type, &is_supported);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (!is_supported) {
        rc = YAKSA_ERR__NOT_SUPPORTED;
        goto fn_exit;
    }

    yaksuri_subreq_s *subreq;
    subreq = (yaksuri_subreq_s *) malloc(sizeof(yaksuri_subreq_s));

    int (*pupfn) (yaksuri_gpudriver_id_e id, const void *inbuf, void *outbuf, uintptr_t count,
                  struct yaksi_type_s * type, struct yaksi_info_s * info, int device);
    if (backend->optype == YAKSURI_OPTYPE__PACK) {
        pupfn = ipack;
    } else {
        pupfn = iunpack;
    }

    uintptr_t threshold;
    if (backend->optype == YAKSURI_OPTYPE__PACK) {
        threshold = yaksuri_global.gpudriver[id].info->get_iov_pack_threshold(info);
    } else {
        threshold = yaksuri_global.gpudriver[id].info->get_iov_unpack_threshold(info);
    }

    if (backend->inattr.type == YAKSUR_PTR_TYPE__GPU &&
        backend->outattr.type == YAKSUR_PTR_TYPE__GPU &&
        backend->inattr.device == backend->outattr.device) {

        subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
        rc = pupfn(id, inbuf, outbuf, count, type, info, backend->inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);
        event_create(id, backend->inattr.device, &subreq->u.single.event);
        event_record(id, subreq->u.single.event);

        goto enqueue_subreq;
    }

    if (backend->inattr.type == YAKSUR_PTR_TYPE__GPU &&
        backend->outattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST &&
        (type->is_contig || type->size / type->num_contig >= threshold)) {

        subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
        rc = pupfn(id, inbuf, outbuf, count, type, info, backend->inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);
        event_create(id, backend->inattr.device, &subreq->u.single.event);
        event_record(id, subreq->u.single.event);

        goto enqueue_subreq;
    }

    if (backend->inattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST &&
        backend->outattr.type == YAKSUR_PTR_TYPE__GPU &&
        (type->is_contig || type->size / type->num_contig >= threshold)) {

        subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
        rc = pupfn(id, inbuf, outbuf, count, type, info, backend->outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);
        event_create(id, backend->outattr.device, &subreq->u.single.event);
        event_record(id, subreq->u.single.event);

        goto enqueue_subreq;
    }

    /* we can only take on types where at least one count of the type
     * fits into our temporary buffers. */
    if (type->size > YAKSURI_TMPBUF_EL_SIZE) {
        rc = YAKSA_ERR__NOT_SUPPORTED;
        free(subreq);
        goto fn_exit;
    }

    subreq->kind = YAKSURI_SUBREQ_KIND__MULTI_CHUNK;

    subreq->u.multiple.inbuf = inbuf;
    subreq->u.multiple.outbuf = outbuf;
    subreq->u.multiple.count = count;
    subreq->u.multiple.type = type;
    subreq->u.multiple.issued_count = 0;
    subreq->u.multiple.chunks = NULL;

    if (backend->optype == YAKSURI_OPTYPE__PACK) {
        if (backend->inattr.type == YAKSUR_PTR_TYPE__GPU &&
            backend->outattr.type == YAKSUR_PTR_TYPE__GPU) {
            subreq->u.multiple.acquire = pack_d2d_acquire;
            subreq->u.multiple.release = simple_release;
        } else if (backend->inattr.type == YAKSUR_PTR_TYPE__GPU) {
            if (backend->outattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST) {
                subreq->u.multiple.acquire = pack_d2rh_acquire;
                subreq->u.multiple.release = simple_release;
            } else {
                subreq->u.multiple.acquire = pack_d2urh_acquire;
                subreq->u.multiple.release = pack_d2urh_release;
            }
        } else if (backend->outattr.type == YAKSUR_PTR_TYPE__GPU) {
            subreq->u.multiple.acquire = pack_h2d_acquire;
            subreq->u.multiple.release = simple_release;
        }
    } else {
        if (backend->inattr.type == YAKSUR_PTR_TYPE__GPU &&
            backend->outattr.type == YAKSUR_PTR_TYPE__GPU) {
            subreq->u.multiple.acquire = unpack_d2d_acquire;
            subreq->u.multiple.release = simple_release;
        } else if (backend->inattr.type == YAKSUR_PTR_TYPE__GPU) {
            subreq->u.multiple.acquire = unpack_d2h_acquire;
            subreq->u.multiple.release = unpack_d2h_release;
        } else if (backend->outattr.type == YAKSUR_PTR_TYPE__GPU) {
            if (backend->inattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST) {
                subreq->u.multiple.acquire = unpack_rh2d_acquire;
                subreq->u.multiple.release = simple_release;
            } else {
                subreq->u.multiple.acquire = unpack_urh2d_acquire;
                subreq->u.multiple.release = simple_release;
            }
        }
    }

  enqueue_subreq:
    pthread_mutex_lock(&progress_mutex);
    DL_APPEND(backend->subreqs, subreq);

    /* if the request is not in our pending list, add it */
    yaksuri_request_s *req;
    HASH_FIND_PTR(pending_reqs, &request, req);
    if (req == NULL) {
        HASH_ADD_PTR(pending_reqs, request, backend);
        yaksu_atomic_incr(&request->cc);
    }
    pthread_mutex_unlock(&progress_mutex);

    rc = yaksuri_progress_poke();
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_progress_poke(void)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id;

    /* A progress poke is in two steps.  In the first step, we check
     * for event completions, finish any post-processing and retire
     * any temporary resources.  In the second steps, we issue out any
     * pending operations. */

    pthread_mutex_lock(&progress_mutex);

    yaksi_type_s *byte_type;
    rc = yaksi_type_get(YAKSA_TYPE__BYTE, &byte_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /**********************************************************************/
    /* Step 1: Check for completions */
    /**********************************************************************/
    yaksuri_request_s *backend, *tmp;
    HASH_ITER(hh, pending_reqs, backend, tmp) {
        id = backend->gpudriver_id;
        assert(backend->subreqs);

        yaksuri_subreq_s *subreq, *tmp2;
        DL_FOREACH_SAFE(backend->subreqs, subreq, tmp2) {
            if (subreq->kind == YAKSURI_SUBREQ_KIND__SINGLE_CHUNK) {
                int completed;
                rc = event_query(id, subreq->u.single.event, &completed);
                YAKSU_ERR_CHECK(rc, fn_fail);

                if (!completed)
                    continue;

                rc = event_destroy(id, subreq->u.single.event);
                YAKSU_ERR_CHECK(rc, fn_fail);

                DL_DELETE(backend->subreqs, subreq);
                free(subreq);
                if (backend->subreqs == NULL) {
                    HASH_DEL(pending_reqs, backend);
                    yaksu_atomic_decr(&backend->request->cc);
                }
            } else {
                yaksuri_subreq_chunk_s *chunk, *tmp3;
                DL_FOREACH_SAFE(subreq->u.multiple.chunks, chunk, tmp3) {
                    int completed;
                    rc = event_query(id, chunk->event, &completed);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    if (!completed)
                        continue;

                    rc = subreq->u.multiple.release(backend, subreq, chunk);
                    YAKSU_ERR_CHECK(rc, fn_fail);
                }
            }
        }
    }

    /**********************************************************************/
    /* Step 2: Issue new operations */
    /**********************************************************************/
    HASH_ITER(hh, pending_reqs, backend, tmp) {
        id = backend->gpudriver_id;
        assert(backend->subreqs);

        yaksuri_subreq_s *subreq, *tmp2;
        DL_FOREACH_SAFE(backend->subreqs, subreq, tmp2) {
            if (subreq->kind == YAKSURI_SUBREQ_KIND__SINGLE_CHUNK)
                continue;

            while (subreq->u.multiple.issued_count < subreq->u.multiple.count) {
                yaksuri_subreq_chunk_s *chunk;

                rc = subreq->u.multiple.acquire(backend, subreq, &chunk);
                YAKSU_ERR_CHECK(rc, fn_fail);

                if (chunk == NULL)
                    goto fn_exit;

                subreq->u.multiple.issued_count += chunk->count;
            }
        }
    }

  fn_exit:
    pthread_mutex_unlock(&progress_mutex);
    return rc;
  fn_fail:
    goto fn_exit;
}
