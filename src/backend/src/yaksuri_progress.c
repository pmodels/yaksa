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

static yaksi_type_s *get_base_type(yaksi_type_s * type)
{
    switch (type->kind) {
        case YAKSI_TYPE_KIND__BUILTIN:
            return type;

        case YAKSI_TYPE_KIND__CONTIG:
            return get_base_type(type->u.contig.child);

        case YAKSI_TYPE_KIND__RESIZED:
            return get_base_type(type->u.resized.child);

        case YAKSI_TYPE_KIND__HVECTOR:
            return get_base_type(type->u.hvector.child);

        case YAKSI_TYPE_KIND__BLKHINDX:
            return get_base_type(type->u.blkhindx.child);

        case YAKSI_TYPE_KIND__HINDEXED:
            return get_base_type(type->u.hindexed.child);

        case YAKSI_TYPE_KIND__SUBARRAY:
            return get_base_type(type->u.subarray.primary);

        default:
            return NULL;
    }
}

static int icopy(yaksuri_gpudriver_id_e id, const void *inbuf, void *outbuf, uintptr_t count,
                 yaksi_type_s * type, yaksi_info_s * info, yaksa_op_t op, int device)
{
    int rc = YAKSA_SUCCESS;
    yaksi_type_s *base_type = get_base_type(type);

    rc = yaksuri_global.gpudriver[id].hooks->ipack(inbuf, outbuf,
                                                   count * type->size / base_type->size,
                                                   base_type, info, op, device);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int ipack(yaksuri_gpudriver_id_e id, const void *inbuf, void *outbuf, uintptr_t count,
                 yaksi_type_s * type, yaksi_info_s * info, yaksa_op_t op, int device)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].hooks->ipack(inbuf, outbuf, count, type, info, op, device);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int iunpack(yaksuri_gpudriver_id_e id, const void *inbuf, void *outbuf, uintptr_t count,
                   yaksi_type_s * type, yaksi_info_s * info, yaksa_op_t op, int device)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].hooks->iunpack(inbuf, outbuf, count, type, info, op, device);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int check_p2p_comm(yaksuri_gpudriver_id_e id, int indev, int outdev, bool * is_enabled)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].hooks->check_p2p_comm(indev, outdev, is_enabled);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int event_record(yaksuri_gpudriver_id_e id, int device, void **event)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].hooks->event_record(device, event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int event_query(yaksuri_gpudriver_id_e id, void *event, int *completed)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].hooks->event_query(event, completed);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int add_dependency(yaksuri_gpudriver_id_e id, int device1, int device2)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_global.gpudriver[id].hooks->add_dependency(device1, device2);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int alloc_chunk(yaksuri_gpudriver_id_e id, yaksuri_request_s * reqpriv,
                       yaksuri_subreq_s * subreq, int num_tmpbufs, int *devices,
                       yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_tmpbuf_s tmpbufs[YAKSURI_SUBREQ_CHUNK_MAX_TMPBUFS];

    assert(subreq);
    assert(subreq->kind == YAKSURI_SUBREQ_KIND__MULTI_CHUNK);

    *chunk = NULL;

    for (int i = 0; i < num_tmpbufs; i++) {
        void *buf;
        if (devices[i] >= 0) {
            rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].device[devices[i]],
                                              &buf);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else {
            rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].host, &buf);
            YAKSU_ERR_CHECK(rc, fn_fail);
        }

        if (buf == NULL) {
            for (int j = 0; j < i; j++) {
                rc = yaksu_buffer_pool_elem_free(tmpbufs[j].pool, tmpbufs[j].buf);
                YAKSU_ERR_CHECK(rc, fn_fail);
            }
            goto fn_exit;
        } else {
            tmpbufs[i].buf = buf;
            if (devices[i] >= 0) {
                tmpbufs[i].pool = yaksuri_global.gpudriver[id].device[devices[i]];
            } else {
                tmpbufs[i].pool = yaksuri_global.gpudriver[id].host;
            }
        }
    }

    /* allocate the chunk */
    *chunk = (yaksuri_subreq_chunk_s *) malloc(sizeof(yaksuri_subreq_chunk_s));

    (*chunk)->count_offset = subreq->u.multiple.issued_count;
    uintptr_t count_per_chunk;
    count_per_chunk = YAKSURI_TMPBUF_EL_SIZE / subreq->u.multiple.type->size;
    if ((*chunk)->count_offset + count_per_chunk <= subreq->u.multiple.count) {
        (*chunk)->count = count_per_chunk;
    } else {
        (*chunk)->count = subreq->u.multiple.count - (*chunk)->count_offset;
    }

    (*chunk)->num_tmpbufs = num_tmpbufs;
    memcpy((*chunk)->tmpbufs, tmpbufs, YAKSURI_SUBREQ_CHUNK_MAX_TMPBUFS * sizeof(yaksuri_tmpbuf_s));
    (*chunk)->event = NULL;

    DL_APPEND(subreq->u.multiple.chunks, (*chunk));

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int simple_release(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                          yaksuri_subreq_chunk_s * chunk)
{
    int rc = YAKSA_SUCCESS;

    /* cleanup */
    for (int i = 0; i < chunk->num_tmpbufs; i++) {
        rc = yaksu_buffer_pool_elem_free(chunk->tmpbufs[i].pool, chunk->tmpbufs[i].buf);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    DL_DELETE(subreq->u.multiple.chunks, chunk);
    free(chunk);

    if (subreq->u.multiple.chunks == NULL) {
        DL_DELETE(reqpriv->subreqs, subreq);
        yaksi_type_free(subreq->u.multiple.type);
        free(subreq);
    }
    if (reqpriv->subreqs == NULL) {
        HASH_DEL(pending_reqs, reqpriv);
        yaksu_atomic_decr(&reqpriv->request->cc);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_d2d_p2p_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                                yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;
    yaksa_op_t op = subreq->u.multiple.op;

    assert(reqpriv->request->backend.inattr.device != reqpriv->request->backend.outattr.device);

    *chunk = NULL;

    if (op == YAKSA_OP__REPLACE) {
        int devices[] = { reqpriv->request->backend.inattr.device };
        rc = alloc_chunk(id, reqpriv, subreq, 1, devices, chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        int devices[] = { reqpriv->request->backend.inattr.device,
            reqpriv->request->backend.outattr.device
        };
        rc = alloc_chunk(id, reqpriv, subreq, 2, devices, chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    if (*chunk == NULL)
        goto fn_exit;

    void *src_d_buf, *dst_d_buf;
    src_d_buf = (*chunk)->tmpbufs[0].buf;
    dst_d_buf = (*chunk)->tmpbufs[1].buf;

    const char *sbuf;
    char *dbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;
    dbuf =
        (char *) subreq->u.multiple.outbuf + (*chunk)->count_offset * subreq->u.multiple.type->size;

    rc = ipack(id, sbuf, src_d_buf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
               YAKSA_OP__REPLACE, reqpriv->request->backend.inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (op == YAKSA_OP__REPLACE) {
        rc = icopy(id, src_d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type,
                   reqpriv->info, YAKSA_OP__REPLACE, reqpriv->request->backend.inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_record(id, reqpriv->request->backend.inattr.device, &(*chunk)->event);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        rc = icopy(id, src_d_buf, dst_d_buf, (*chunk)->count, subreq->u.multiple.type,
                   reqpriv->info, YAKSA_OP__REPLACE, reqpriv->request->backend.inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = add_dependency(id, reqpriv->request->backend.inattr.device,
                            reqpriv->request->backend.outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = icopy(id, dst_d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type,
                   reqpriv->info, op, reqpriv->request->backend.outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = event_record(id, reqpriv->request->backend.outattr.device, &(*chunk)->event);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_d2d_nop2p_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                                  yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;
    yaksa_op_t op = subreq->u.multiple.op;

    assert(reqpriv->request->backend.inattr.device != reqpriv->request->backend.outattr.device);

    *chunk = NULL;

    if (op == YAKSA_OP__REPLACE) {
        int devices[] = { reqpriv->request->backend.inattr.device, -1 };
        rc = alloc_chunk(id, reqpriv, subreq, 2, devices, chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        int devices[] = { reqpriv->request->backend.inattr.device, -1,
            reqpriv->request->backend.outattr.device
        };
        rc = alloc_chunk(id, reqpriv, subreq, 3, devices, chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    if (*chunk == NULL)
        goto fn_exit;

    void *src_d_buf, *rh_buf, *dst_d_buf;
    src_d_buf = (*chunk)->tmpbufs[0].buf;
    rh_buf = (*chunk)->tmpbufs[1].buf;
    dst_d_buf = (*chunk)->tmpbufs[2].buf;

    const char *sbuf;
    char *dbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;
    dbuf =
        (char *) subreq->u.multiple.outbuf + (*chunk)->count_offset * subreq->u.multiple.type->size;

    rc = ipack(id, sbuf, src_d_buf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
               YAKSA_OP__REPLACE, reqpriv->request->backend.inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = icopy(id, src_d_buf, rh_buf, (*chunk)->count, subreq->u.multiple.type,
               reqpriv->info, YAKSA_OP__REPLACE, reqpriv->request->backend.inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = add_dependency(id, reqpriv->request->backend.inattr.device,
                        reqpriv->request->backend.outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (op == YAKSA_OP__REPLACE) {
        rc = icopy(id, rh_buf, dbuf, (*chunk)->count, subreq->u.multiple.type,
                   reqpriv->info, YAKSA_OP__REPLACE, reqpriv->request->backend.outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        rc = icopy(id, rh_buf, dst_d_buf, (*chunk)->count, subreq->u.multiple.type,
                   reqpriv->info, YAKSA_OP__REPLACE, reqpriv->request->backend.outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = icopy(id, dst_d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type,
                   reqpriv->info, op, reqpriv->request->backend.outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    rc = event_record(id, reqpriv->request->backend.outattr.device, &(*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_d2rh_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                             yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;
    yaksa_op_t op = subreq->u.multiple.op;

    *chunk = NULL;

    if (op == YAKSA_OP__REPLACE) {
        int devices[] = { reqpriv->request->backend.inattr.device };
        rc = alloc_chunk(id, reqpriv, subreq, 1, devices, chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        int devices[] = { reqpriv->request->backend.inattr.device, -1 };
        rc = alloc_chunk(id, reqpriv, subreq, 2, devices, chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    if (*chunk == NULL)
        goto fn_exit;

    void *d_buf, *rh_buf;
    d_buf = (*chunk)->tmpbufs[0].buf;
    rh_buf = (*chunk)->tmpbufs[1].buf;

    const char *sbuf;
    char *dbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;
    dbuf = (char *) subreq->u.multiple.outbuf + (*chunk)->count_offset *
        subreq->u.multiple.type->size;

    rc = ipack(id, sbuf, d_buf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
               YAKSA_OP__REPLACE, reqpriv->request->backend.inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (op == YAKSA_OP__REPLACE) {
        rc = icopy(id, d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
                   YAKSA_OP__REPLACE, reqpriv->request->backend.inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        rc = icopy(id, d_buf, rh_buf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
                   YAKSA_OP__REPLACE, reqpriv->request->backend.inattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    rc = event_record(id, reqpriv->request->backend.inattr.device, &(*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_d2rh_release(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                             yaksuri_subreq_chunk_s * chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksa_op_t op = subreq->u.multiple.op;
    yaksi_type_s *type = subreq->u.multiple.type;

    if (op != YAKSA_OP__REPLACE) {
        yaksi_type_s *base_type = get_base_type(type);
        char *dbuf = (char *) subreq->u.multiple.outbuf + chunk->count_offset *
            subreq->u.multiple.type->size;

        rc = yaksuri_seq_ipack(chunk->tmpbufs[1].buf, dbuf,
                               chunk->count * type->size / base_type->size, base_type,
                               reqpriv->info, subreq->u.multiple.op);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    rc = simple_release(reqpriv, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_d2urh_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                              yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;

    *chunk = NULL;

    int devices[] = { reqpriv->request->backend.inattr.device, -1 };
    rc = alloc_chunk(id, reqpriv, subreq, 2, devices, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (*chunk == NULL)
        goto fn_exit;

    void *d_buf, *rh_buf;
    d_buf = (*chunk)->tmpbufs[0].buf;
    rh_buf = (*chunk)->tmpbufs[1].buf;

    const char *sbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;

    rc = ipack(id, sbuf, d_buf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
               YAKSA_OP__REPLACE, reqpriv->request->backend.inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = icopy(id, d_buf, rh_buf, (*chunk)->count, subreq->u.multiple.type,
               reqpriv->info, YAKSA_OP__REPLACE, reqpriv->request->backend.inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = event_record(id, reqpriv->request->backend.inattr.device, &(*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_d2urh_release(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                              yaksuri_subreq_chunk_s * chunk)
{
    int rc = YAKSA_SUCCESS;
    char *dbuf = (char *) subreq->u.multiple.outbuf + chunk->count_offset *
        subreq->u.multiple.type->size;
    yaksi_type_s *type = subreq->u.multiple.type;
    yaksi_type_s *base_type = get_base_type(type);

    rc = yaksuri_seq_ipack(chunk->tmpbufs[1].buf, dbuf,
                           chunk->count * type->size / base_type->size, base_type,
                           reqpriv->info, subreq->u.multiple.op);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = simple_release(reqpriv, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int pack_h2d_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                            yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;
    yaksa_op_t op = subreq->u.multiple.op;

    *chunk = NULL;

    if (op == YAKSA_OP__REPLACE) {
        int devices[] = { -1 };
        rc = alloc_chunk(id, reqpriv, subreq, 1, devices, chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        int devices[] = { -1, reqpriv->request->backend.outattr.device };
        rc = alloc_chunk(id, reqpriv, subreq, 2, devices, chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    if (*chunk == NULL)
        goto fn_exit;

    void *rh_buf, *d_buf;
    rh_buf = (*chunk)->tmpbufs[0].buf;
    d_buf = (*chunk)->tmpbufs[1].buf;

    const char *sbuf;
    char *dbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;
    dbuf = (char *) subreq->u.multiple.outbuf + (*chunk)->count_offset *
        subreq->u.multiple.type->size;

    rc = yaksuri_seq_ipack(sbuf, rh_buf, (*chunk)->count, subreq->u.multiple.type,
                           reqpriv->info, YAKSA_OP__REPLACE);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (op == YAKSA_OP__REPLACE) {
        rc = icopy(id, rh_buf, dbuf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
                   YAKSA_OP__REPLACE, reqpriv->request->backend.outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        rc = icopy(id, rh_buf, d_buf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
                   YAKSA_OP__REPLACE, reqpriv->request->backend.outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = icopy(id, d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
                   op, reqpriv->request->backend.outattr.device);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    rc = event_record(id, reqpriv->request->backend.outattr.device, &(*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int unpack_d2d_p2p_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                                  yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;

    assert(reqpriv->request->backend.inattr.device != reqpriv->request->backend.outattr.device);

    *chunk = NULL;

    int devices[] = { reqpriv->request->backend.outattr.device };
    rc = alloc_chunk(id, reqpriv, subreq, 1, devices, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (*chunk == NULL)
        goto fn_exit;

    void *d_buf;
    d_buf = (*chunk)->tmpbufs[0].buf;

    const char *sbuf;
    char *dbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->size;
    dbuf = (char *) subreq->u.multiple.outbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;

    rc = icopy(id, sbuf, d_buf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
               YAKSA_OP__REPLACE, reqpriv->request->backend.inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = add_dependency(id, reqpriv->request->backend.inattr.device,
                        reqpriv->request->backend.outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = iunpack(id, d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
                 subreq->u.multiple.op, reqpriv->request->backend.outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = event_record(id, reqpriv->request->backend.outattr.device, &(*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int unpack_d2d_nop2p_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                                    yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;

    assert(reqpriv->request->backend.inattr.device != reqpriv->request->backend.outattr.device);

    *chunk = NULL;

    int devices[] = { reqpriv->request->backend.outattr.device, -1 };
    rc = alloc_chunk(id, reqpriv, subreq, 2, devices, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (*chunk == NULL)
        goto fn_exit;

    void *d_buf, *rh_buf;
    d_buf = (*chunk)->tmpbufs[0].buf;
    rh_buf = (*chunk)->tmpbufs[1].buf;

    const char *sbuf;
    char *dbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->size;
    dbuf = (char *) subreq->u.multiple.outbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;

    rc = icopy(id, sbuf, rh_buf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
               YAKSA_OP__REPLACE, reqpriv->request->backend.inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = add_dependency(id, reqpriv->request->backend.inattr.device,
                        reqpriv->request->backend.outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = icopy(id, rh_buf, d_buf, (*chunk)->count, subreq->u.multiple.type,
               reqpriv->info, YAKSA_OP__REPLACE, reqpriv->request->backend.outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = iunpack(id, d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type,
                 reqpriv->info, subreq->u.multiple.op, reqpriv->request->backend.outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = event_record(id, reqpriv->request->backend.outattr.device, &(*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int unpack_rh2d_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                               yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;

    *chunk = NULL;

    int devices[] = { reqpriv->request->backend.outattr.device };
    rc = alloc_chunk(id, reqpriv, subreq, 1, devices, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (*chunk == NULL)
        goto fn_exit;

    void *d_buf;
    d_buf = (*chunk)->tmpbufs[0].buf;

    const char *sbuf;
    char *dbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->size;
    dbuf = (char *) subreq->u.multiple.outbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;

    rc = icopy(id, sbuf, d_buf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
               YAKSA_OP__REPLACE, reqpriv->request->backend.outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = iunpack(id, d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
                 subreq->u.multiple.op, reqpriv->request->backend.outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = event_record(id, reqpriv->request->backend.outattr.device, &(*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int unpack_urh2d_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                                yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;

    *chunk = NULL;

    int devices[] = { reqpriv->request->backend.outattr.device, -1 };
    rc = alloc_chunk(id, reqpriv, subreq, 2, devices, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (*chunk == NULL)
        goto fn_exit;

    void *d_buf, *rh_buf;
    d_buf = (*chunk)->tmpbufs[0].buf;
    rh_buf = (*chunk)->tmpbufs[1].buf;

    const char *sbuf;
    char *dbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->size;
    dbuf = (char *) subreq->u.multiple.outbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->extent;

    yaksi_type_s *byte_type;
    rc = yaksi_type_get(YAKSA_TYPE__BYTE, &byte_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksuri_seq_ipack(sbuf, rh_buf, (*chunk)->count * subreq->u.multiple.type->size,
                           byte_type, reqpriv->info, YAKSA_OP__REPLACE);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = icopy(id, rh_buf, d_buf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
               YAKSA_OP__REPLACE, reqpriv->request->backend.outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = iunpack(id, d_buf, dbuf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
                 subreq->u.multiple.op, reqpriv->request->backend.outattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = event_record(id, reqpriv->request->backend.outattr.device, &(*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int unpack_d2h_acquire(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                              yaksuri_subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;

    *chunk = NULL;

    int devices[] = { -1 };
    rc = alloc_chunk(id, reqpriv, subreq, 1, devices, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (*chunk == NULL)
        goto fn_exit;

    void *rh_buf;
    rh_buf = (*chunk)->tmpbufs[0].buf;

    const char *sbuf;
    sbuf = (const char *) subreq->u.multiple.inbuf +
        (*chunk)->count_offset * subreq->u.multiple.type->size;

    rc = icopy(id, sbuf, rh_buf, (*chunk)->count, subreq->u.multiple.type, reqpriv->info,
               YAKSA_OP__REPLACE, reqpriv->request->backend.inattr.device);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = event_record(id, reqpriv->request->backend.inattr.device, &(*chunk)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int unpack_d2h_release(yaksuri_request_s * reqpriv, yaksuri_subreq_s * subreq,
                              yaksuri_subreq_chunk_s * chunk)
{
    int rc = YAKSA_SUCCESS;

    char *dbuf;
    dbuf = (char *) subreq->u.multiple.outbuf + chunk->count_offset *
        subreq->u.multiple.type->extent;
    rc = yaksuri_seq_iunpack(chunk->tmpbufs[0].buf, dbuf, chunk->count, subreq->u.multiple.type,
                             reqpriv->info, subreq->u.multiple.op);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = simple_release(reqpriv, subreq, chunk);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_progress_enqueue(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                             yaksi_info_s * info, yaksa_op_t op, yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_request_s *reqpriv = (yaksuri_request_s *) request->backend.priv;
    yaksuri_gpudriver_id_e id = reqpriv->gpudriver_id;

    assert(yaksuri_global.gpudriver[id].hooks);
    assert(request->backend.inattr.type == YAKSUR_PTR_TYPE__GPU ||
           request->backend.outattr.type == YAKSUR_PTR_TYPE__GPU);

    reqpriv->info = info;

    /* if the GPU reqpriv cannot support this type, return */
    bool is_supported;
    rc = yaksuri_global.gpudriver[id].hooks->pup_is_supported(type, op, &is_supported);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (!is_supported) {
        rc = YAKSA_ERR__NOT_SUPPORTED;
        goto fn_exit;
    }

    yaksuri_subreq_s *subreq;
    subreq = (yaksuri_subreq_s *) malloc(sizeof(yaksuri_subreq_s));


    uintptr_t threshold;
    if (reqpriv->optype == YAKSURI_OPTYPE__PACK) {
        threshold = yaksuri_global.gpudriver[id].hooks->get_iov_pack_threshold(info);

        if (request->backend.inattr.type == YAKSUR_PTR_TYPE__GPU) {
            if (request->backend.outattr.type == YAKSUR_PTR_TYPE__GPU) {
                if (request->backend.inattr.device == request->backend.outattr.device) {
                    subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
                    rc = ipack(id, inbuf, outbuf, count, type, info, op,
                               request->backend.inattr.device);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    rc = event_record(id, request->backend.inattr.device, &subreq->u.single.event);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    goto enqueue_subreq;
                } else {
                    bool is_enabled;
                    rc = check_p2p_comm(id, reqpriv->request->backend.inattr.device,
                                        reqpriv->request->backend.outattr.device, &is_enabled);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    if (is_enabled) {
                        subreq->u.multiple.acquire = pack_d2d_p2p_acquire;
                    } else {
                        subreq->u.multiple.acquire = pack_d2d_nop2p_acquire;
                    }
                    subreq->u.multiple.release = simple_release;

                    goto multi_chunk;
                }
            } else if (request->backend.outattr.type == YAKSUR_PTR_TYPE__MANAGED) {
                if (request->backend.outattr.device == -1 ||
                    request->backend.inattr.device == request->backend.outattr.device) {
                    subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
                    rc = ipack(id, inbuf, outbuf, count, type, info, op,
                               request->backend.inattr.device);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    rc = event_record(id, request->backend.inattr.device, &subreq->u.single.event);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    goto enqueue_subreq;
                } else {
                    subreq->u.multiple.acquire = pack_d2urh_acquire;
                    subreq->u.multiple.release = pack_d2urh_release;
                    goto multi_chunk;
                }
            } else if (request->backend.outattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST) {
                if (op == YAKSA_OP__REPLACE) {
                    if (type->is_contig || type->size / type->num_contig >= threshold) {
                        subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
                        rc = ipack(id, inbuf, outbuf, count, type, info, op,
                                   request->backend.inattr.device);
                        YAKSU_ERR_CHECK(rc, fn_fail);

                        rc = event_record(id, request->backend.inattr.device,
                                          &subreq->u.single.event);
                        YAKSU_ERR_CHECK(rc, fn_fail);

                        goto enqueue_subreq;
                    } else {
                        subreq->u.multiple.acquire = pack_d2rh_acquire;
                        subreq->u.multiple.release = pack_d2rh_release;
                        goto multi_chunk;
                    }
                } else {
                    subreq->u.multiple.acquire = pack_d2rh_acquire;
                    subreq->u.multiple.release = pack_d2rh_release;
                    goto multi_chunk;
                }
            } else {
                subreq->u.multiple.acquire = pack_d2urh_acquire;
                subreq->u.multiple.release = pack_d2urh_release;
                goto multi_chunk;
            }
        } else if (request->backend.inattr.type == YAKSUR_PTR_TYPE__MANAGED) {
            if (request->backend.inattr.device == -1 ||
                request->backend.inattr.device == request->backend.outattr.device) {
                subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
                rc = ipack(id, inbuf, outbuf, count, type, info, op,
                           request->backend.outattr.device);
                YAKSU_ERR_CHECK(rc, fn_fail);

                rc = event_record(id, request->backend.outattr.device, &subreq->u.single.event);
                YAKSU_ERR_CHECK(rc, fn_fail);

                goto enqueue_subreq;
            } else {
                subreq->u.multiple.acquire = pack_h2d_acquire;
                subreq->u.multiple.release = simple_release;
                goto multi_chunk;
            }
        } else if (request->backend.inattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST) {
            if (op == YAKSA_OP__REPLACE) {
                if (type->is_contig || type->size / type->num_contig >= threshold) {
                    subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
                    rc = ipack(id, inbuf, outbuf, count, type, info, op,
                               request->backend.outattr.device);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    rc = event_record(id, request->backend.outattr.device, &subreq->u.single.event);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    goto enqueue_subreq;
                } else {
                    subreq->u.multiple.acquire = pack_h2d_acquire;
                    subreq->u.multiple.release = simple_release;
                    goto multi_chunk;
                }
            } else {
                subreq->u.multiple.acquire = pack_h2d_acquire;
                subreq->u.multiple.release = simple_release;
                goto multi_chunk;
            }
        } else {
            subreq->u.multiple.acquire = pack_h2d_acquire;
            subreq->u.multiple.release = simple_release;
            goto multi_chunk;
        }
    } else {
        threshold = yaksuri_global.gpudriver[id].hooks->get_iov_unpack_threshold(info);

        if (request->backend.inattr.type == YAKSUR_PTR_TYPE__GPU) {
            if (request->backend.outattr.type == YAKSUR_PTR_TYPE__GPU) {
                if (request->backend.inattr.device == request->backend.outattr.device) {
                    subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
                    rc = iunpack(id, inbuf, outbuf, count, type, info, op,
                                 request->backend.inattr.device);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    rc = event_record(id, request->backend.inattr.device, &subreq->u.single.event);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    goto enqueue_subreq;
                } else {
                    bool is_enabled;
                    rc = check_p2p_comm(id, reqpriv->request->backend.inattr.device,
                                        reqpriv->request->backend.outattr.device, &is_enabled);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    if (is_enabled) {
                        subreq->u.multiple.acquire = unpack_d2d_p2p_acquire;
                    } else {
                        subreq->u.multiple.acquire = unpack_d2d_nop2p_acquire;
                    }
                    subreq->u.multiple.release = simple_release;

                    goto multi_chunk;
                }
            } else if (request->backend.outattr.type == YAKSUR_PTR_TYPE__MANAGED) {
                if (request->backend.outattr.device == -1 ||
                    request->backend.inattr.device == request->backend.outattr.device) {
                    subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
                    rc = iunpack(id, inbuf, outbuf, count, type, info, op,
                                 request->backend.inattr.device);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    rc = event_record(id, request->backend.inattr.device, &subreq->u.single.event);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    goto enqueue_subreq;
                } else {
                    subreq->u.multiple.acquire = unpack_d2h_acquire;
                    subreq->u.multiple.release = unpack_d2h_release;
                    goto multi_chunk;
                }
            } else if (request->backend.outattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST) {
                if (op == YAKSA_OP__REPLACE) {
                    if (type->is_contig || type->size / type->num_contig >= threshold) {
                        subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
                        rc = iunpack(id, inbuf, outbuf, count, type, info, op,
                                     request->backend.inattr.device);
                        YAKSU_ERR_CHECK(rc, fn_fail);

                        rc = event_record(id, request->backend.inattr.device,
                                          &subreq->u.single.event);
                        YAKSU_ERR_CHECK(rc, fn_fail);

                        goto enqueue_subreq;
                    } else {
                        subreq->u.multiple.acquire = unpack_d2h_acquire;
                        subreq->u.multiple.release = unpack_d2h_release;
                        goto multi_chunk;
                    }
                } else {
                    subreq->u.multiple.acquire = unpack_d2h_acquire;
                    subreq->u.multiple.release = unpack_d2h_release;
                    goto multi_chunk;
                }
            } else {
                subreq->u.multiple.acquire = unpack_d2h_acquire;
                subreq->u.multiple.release = unpack_d2h_release;
                goto multi_chunk;
            }
        } else if (request->backend.inattr.type == YAKSUR_PTR_TYPE__MANAGED) {
            if (request->backend.inattr.device == -1 ||
                request->backend.inattr.device == request->backend.outattr.device) {
                subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
                rc = iunpack(id, inbuf, outbuf, count, type, info, op,
                             request->backend.outattr.device);
                YAKSU_ERR_CHECK(rc, fn_fail);

                rc = event_record(id, request->backend.outattr.device, &subreq->u.single.event);
                YAKSU_ERR_CHECK(rc, fn_fail);

                goto enqueue_subreq;
            } else {
                subreq->u.multiple.acquire = unpack_urh2d_acquire;
                subreq->u.multiple.release = simple_release;
                goto multi_chunk;
            }
        } else if (request->backend.inattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST) {
            if (op == YAKSA_OP__REPLACE) {
                if (type->is_contig || type->size / type->num_contig >= threshold) {
                    subreq->kind = YAKSURI_SUBREQ_KIND__SINGLE_CHUNK;
                    rc = iunpack(id, inbuf, outbuf, count, type, info, op,
                                 request->backend.outattr.device);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    rc = event_record(id, request->backend.outattr.device, &subreq->u.single.event);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    goto enqueue_subreq;
                } else {
                    subreq->u.multiple.acquire = unpack_rh2d_acquire;
                    subreq->u.multiple.release = simple_release;
                    goto multi_chunk;
                }
            } else {
                subreq->u.multiple.acquire = unpack_rh2d_acquire;
                subreq->u.multiple.release = simple_release;
                goto multi_chunk;
            }
        } else {
            subreq->u.multiple.acquire = unpack_urh2d_acquire;
            subreq->u.multiple.release = simple_release;
            goto multi_chunk;
        }
    }

  multi_chunk:
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
    subreq->u.multiple.op = op;
    subreq->u.multiple.issued_count = 0;
    subreq->u.multiple.chunks = NULL;

    yaksu_atomic_incr(&type->refcount);

  enqueue_subreq:
    pthread_mutex_lock(&progress_mutex);
    DL_APPEND(reqpriv->subreqs, subreq);

    /* if the request is not in our pending list, add it */
    yaksuri_request_s *req;
    HASH_FIND_PTR(pending_reqs, &request, req);
    if (req == NULL) {
        HASH_ADD_PTR(pending_reqs, request, reqpriv);
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
    yaksuri_gpudriver_id_e id = YAKSURI_GPUDRIVER_ID__UNSET;

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
    yaksuri_request_s *reqpriv, *tmp;
    HASH_ITER(hh, pending_reqs, reqpriv, tmp) {
        id = reqpriv->gpudriver_id;
        assert(reqpriv->subreqs);

        yaksuri_subreq_s *subreq, *tmp2;
        DL_FOREACH_SAFE(reqpriv->subreqs, subreq, tmp2) {
            if (subreq->kind == YAKSURI_SUBREQ_KIND__SINGLE_CHUNK) {
                int completed;
                rc = event_query(id, subreq->u.single.event, &completed);
                YAKSU_ERR_CHECK(rc, fn_fail);

                if (!completed)
                    continue;

                DL_DELETE(reqpriv->subreqs, subreq);
                free(subreq);
                if (reqpriv->subreqs == NULL) {
                    HASH_DEL(pending_reqs, reqpriv);
                    yaksu_atomic_decr(&reqpriv->request->cc);
                }
            } else {
                yaksuri_subreq_chunk_s *chunk, *tmp3;
                DL_FOREACH_SAFE(subreq->u.multiple.chunks, chunk, tmp3) {
                    int completed;
                    rc = event_query(id, chunk->event, &completed);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    if (!completed)
                        continue;

                    rc = subreq->u.multiple.release(reqpriv, subreq, chunk);
                    YAKSU_ERR_CHECK(rc, fn_fail);
                }
            }
        }
    }

    /**********************************************************************/
    /* Step 2: Issue new operations */
    /**********************************************************************/
    HASH_ITER(hh, pending_reqs, reqpriv, tmp) {
        id = reqpriv->gpudriver_id;
        assert(reqpriv->subreqs);

        yaksuri_subreq_s *subreq, *tmp2;
        DL_FOREACH_SAFE(reqpriv->subreqs, subreq, tmp2) {
            if (subreq->kind == YAKSURI_SUBREQ_KIND__SINGLE_CHUNK)
                continue;

            while (subreq->u.multiple.issued_count < subreq->u.multiple.count) {
                yaksuri_subreq_chunk_s *chunk;

                rc = subreq->u.multiple.acquire(reqpriv, subreq, &chunk);
                YAKSU_ERR_CHECK(rc, fn_fail);

                if (chunk == NULL)
                    goto flush;

                subreq->u.multiple.issued_count += chunk->count;
            }
        }
    }

  flush:
    /* if we issued any operations, call flush_all, so the driver
     * layer can flush the kernels. */
    if (id != YAKSURI_GPUDRIVER_ID__UNSET) {
        rc = yaksuri_global.gpudriver[id].hooks->flush_all();
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

  fn_exit:
    pthread_mutex_unlock(&progress_mutex);
    return rc;
  fn_fail:
    goto fn_exit;
}
