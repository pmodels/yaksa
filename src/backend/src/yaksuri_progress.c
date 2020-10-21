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

typedef struct subreq_chunk_s {
    uintptr_t count_offset;
    uintptr_t count;
    void *gpu_tmpbuf;
    void *host_tmpbuf;
    void *interm_event;
    void *event;

    struct subreq_chunk_s *next;
    struct subreq_chunk_s *prev;
} subreq_chunk_s;

typedef struct progress_elem_s {
    struct {
        const void *inbuf;
        void *outbuf;
        uintptr_t count;
        yaksi_type_s *type;

        uintptr_t completed_count;
        uintptr_t issued_count;
        subreq_chunk_s *chunks;
    } pup;

    yaksi_request_s *request;
    yaksi_info_s *info;

    struct progress_elem_s *next;
    struct progress_elem_s *prev;
} progress_elem_s;

static progress_elem_s *progress_elems = NULL;
static pthread_mutex_t progress_mutex = PTHREAD_MUTEX_INITIALIZER;

#define ELEM_TO_OPTYPE(elem) \
    (((yaksuri_request_s *) elem->request->backend.priv)->optype)
#define ELEM_TO_INATTR_TYPE(elem) \
    (((yaksuri_request_s *) elem->request->backend.priv)->inattr.type)
#define ELEM_TO_OUTATTR_TYPE(elem) \
    (((yaksuri_request_s *) elem->request->backend.priv)->outattr.type)
#define ELEM_TO_INATTR_DEVICE(elem) \
    (((yaksuri_request_s *) elem->request->backend.priv)->inattr.device)
#define ELEM_TO_OUTATTR_DEVICE(elem) \
    (((yaksuri_request_s *) elem->request->backend.priv)->outattr.device)

int yaksuri_progress_enqueue(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                             yaksi_info_s * info, yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_request_s *backend = (yaksuri_request_s *) request->backend.priv;

    /* if we need to go through the progress engine, make sure we only
     * take on types, where at least one count of the type fits into
     * our temporary buffers. */
    if (type->size > YAKSURI_TMPBUF_EL_SIZE) {
        rc = YAKSA_ERR__NOT_SUPPORTED;
        goto fn_exit;
    }

    /* enqueue to the progress engine */
    progress_elem_s *newelem;
    newelem = (progress_elem_s *) malloc(sizeof(progress_elem_s));

    newelem->pup.inbuf = inbuf;
    newelem->pup.outbuf = outbuf;
    newelem->pup.count = count;
    newelem->pup.type = type;
    newelem->pup.completed_count = 0;
    newelem->pup.issued_count = 0;
    newelem->pup.chunks = NULL;
    newelem->request = request;
    newelem->info = info;
    newelem->next = NULL;

    /* enqueue the new element */
    yaksu_atomic_incr(&request->cc);

    pthread_mutex_lock(&progress_mutex);
    DL_APPEND(progress_elems, newelem);
    pthread_mutex_unlock(&progress_mutex);

  fn_exit:
    return rc;
}

static int alloc_chunk(subreq_chunk_s ** chunk)
{
    int rc = YAKSA_SUCCESS;
    progress_elem_s *elem = progress_elems;
    yaksuri_request_s *request_backend = (yaksuri_request_s *) elem->request->backend.priv;
    yaksuri_gpudriver_id_e id = request_backend->gpudriver_id;
    bool need_gpu_tmpbuf = false, need_host_tmpbuf = false;
    int devid = INT_MIN;

    if ((ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__PACK &&
         ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU &&
         ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__UNREGISTERED_HOST) ||
        (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__PACK &&
         ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__REGISTERED_HOST &&
         ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU) ||
        (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__PACK &&
         ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__UNREGISTERED_HOST &&
         ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU) ||
        (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__UNPACK &&
         ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__UNREGISTERED_HOST &&
         ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU) ||
        (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__UNPACK &&
         ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU &&
         ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__REGISTERED_HOST) ||
        (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__UNPACK &&
         ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU &&
         ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__UNREGISTERED_HOST)) {
        need_host_tmpbuf = true;
    }

    if (ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU &&
        ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU) {
        bool is_enabled;
        rc = yaksuri_global.gpudriver[id].info->check_p2p_comm(ELEM_TO_INATTR_DEVICE(elem),
                                                               ELEM_TO_OUTATTR_DEVICE(elem),
                                                               &is_enabled);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (!is_enabled) {
            need_host_tmpbuf = true;
        }
    }

    if ((ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__PACK &&
         ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU &&
         ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__REGISTERED_HOST) ||
        (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__PACK &&
         ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU &&
         ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__UNREGISTERED_HOST) ||
        (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__PACK &&
         ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU &&
         ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU)) {
        need_gpu_tmpbuf = true;
        devid = ELEM_TO_INATTR_DEVICE(elem);
    }

    if ((ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__UNPACK &&
         ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__REGISTERED_HOST &&
         ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU) ||
        (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__UNPACK &&
         ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__UNREGISTERED_HOST &&
         ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU) ||
        (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__UNPACK &&
         ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU &&
         ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU)) {
        need_gpu_tmpbuf = true;
        devid = ELEM_TO_OUTATTR_DEVICE(elem);
    }

    assert(need_host_tmpbuf || need_gpu_tmpbuf);

    /* allocate the chunk */
    *chunk = (subreq_chunk_s *) malloc(sizeof(subreq_chunk_s));

    /* figure out if we actually have enough buffer space */
    if (need_gpu_tmpbuf) {
        rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].device[devid],
                                          &(*chunk)->gpu_tmpbuf);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if ((*chunk)->gpu_tmpbuf == NULL) {
            free(*chunk);
            *chunk = NULL;
            goto fn_exit;
        }
    } else {
        (*chunk)->gpu_tmpbuf = NULL;
    }

    if (need_host_tmpbuf) {
        rc = yaksu_buffer_pool_elem_alloc(yaksuri_global.gpudriver[id].host,
                                          &(*chunk)->host_tmpbuf);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if ((*chunk)->host_tmpbuf == NULL) {
            if (need_gpu_tmpbuf) {
                rc = yaksu_buffer_pool_elem_free(yaksuri_global.gpudriver[id].device[devid],
                                                 (*chunk)->gpu_tmpbuf);
                YAKSU_ERR_CHECK(rc, fn_fail);
            }
            free(*chunk);
            *chunk = NULL;
            goto fn_exit;
        }
    } else {
        (*chunk)->host_tmpbuf = NULL;
    }

    (*chunk)->count_offset = elem->pup.completed_count + elem->pup.issued_count;

    uintptr_t count_per_chunk = YAKSURI_TMPBUF_EL_SIZE / elem->pup.type->size;
    if ((*chunk)->count_offset + count_per_chunk <= elem->pup.count) {
        (*chunk)->count = count_per_chunk;
    } else {
        (*chunk)->count = elem->pup.count - (*chunk)->count_offset;
    }

    (*chunk)->interm_event = NULL;
    (*chunk)->event = NULL;

    DL_APPEND(elem->pup.chunks, *chunk);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int free_chunk(subreq_chunk_s * chunk)
{
    int rc = YAKSA_SUCCESS;
    progress_elem_s *elem = progress_elems;
    yaksuri_request_s *request_backend = (yaksuri_request_s *) elem->request->backend.priv;
    yaksuri_gpudriver_id_e id = request_backend->gpudriver_id;

    if (chunk->interm_event) {
        rc = yaksuri_global.gpudriver[id].info->event_destroy(chunk->interm_event);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    rc = yaksuri_global.gpudriver[id].info->event_destroy(chunk->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* free the device buffer */
    if (chunk->gpu_tmpbuf) {
        int devid;

        if (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__PACK)
            devid = ELEM_TO_INATTR_DEVICE(elem);
        else
            devid = ELEM_TO_OUTATTR_DEVICE(elem);

        rc = yaksu_buffer_pool_elem_free(yaksuri_global.gpudriver[id].device[devid],
                                         chunk->gpu_tmpbuf);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    /* free the host buffer */
    if (chunk->host_tmpbuf) {
        rc = yaksu_buffer_pool_elem_free(yaksuri_global.gpudriver[id].host, chunk->host_tmpbuf);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    DL_DELETE(elem->pup.chunks, chunk);
    free(chunk);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_progress_poke(void)
{
    int rc = YAKSA_SUCCESS;

    /* We only poke the head of the progress engine as all of our
     * operations are independent.  This reduces the complexity of the
     * progress engine and keeps the amount of time we spend in the
     * progress engine small. */

    pthread_mutex_lock(&progress_mutex);

    /* if there's nothing to do, return */
    if (progress_elems == NULL)
        goto fn_exit;

    /* the progress engine has three steps: (1) check for completions
     * and free up any held up resources; (2) if we don't have
     * anything else to do, return; and (3) issue any pending
     * chunks. */
    yaksi_type_s *byte_type;
    rc = yaksi_type_get(YAKSA_TYPE__BYTE, &byte_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    progress_elem_s *elem;
    elem = progress_elems;
    yaksuri_request_s *request_backend;
    request_backend = (yaksuri_request_s *) elem->request->backend.priv;
    yaksuri_gpudriver_id_e id;
    id = request_backend->gpudriver_id;

    /****************************************************************************/
    /* Step 1: Check for completion and free up any held up resources */
    /****************************************************************************/
    subreq_chunk_s *chunk, *tmp;
    DL_FOREACH_SAFE(elem->pup.chunks, chunk, tmp) {
        int completed;
        rc = yaksuri_global.gpudriver[id].info->event_query(chunk->event, &completed);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (completed) {
            if (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__PACK &&
                ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU &&
                ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__UNREGISTERED_HOST) {
                char *dbuf = (char *) elem->pup.outbuf + chunk->count_offset * elem->pup.type->size;
                rc = yaksuri_seq_ipack(chunk->host_tmpbuf, dbuf,
                                       chunk->count * elem->pup.type->size, byte_type, elem->info);
                YAKSU_ERR_CHECK(rc, fn_fail);
            } else if (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__UNPACK &&
                       ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU &&
                       (ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__REGISTERED_HOST ||
                        ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__UNREGISTERED_HOST)) {
                char *dbuf =
                    (char *) elem->pup.outbuf + chunk->count_offset * elem->pup.type->extent;
                rc = yaksuri_seq_iunpack(chunk->host_tmpbuf, dbuf, chunk->count, elem->pup.type,
                                         elem->info);
                YAKSU_ERR_CHECK(rc, fn_fail);
            }

            elem->pup.completed_count += chunk->count;
            elem->pup.issued_count -= chunk->count;

            subreq_chunk_s *tmp = chunk->next;
            rc = free_chunk(chunk);
            YAKSU_ERR_CHECK(rc, fn_fail);
            chunk = tmp;
        } else {
            break;
        }
    }


    /****************************************************************************/
    /* Step 2: If we don't have any more work to do, return */
    /****************************************************************************/
    if (elem->pup.completed_count == elem->pup.count) {
        DL_DELETE(progress_elems, elem);
        yaksu_atomic_decr(&elem->request->cc);
        free(elem);
        goto fn_exit;
    }


    /****************************************************************************/
    /* Step 3: Issue any pending chunks */
    /****************************************************************************/
    while (elem->pup.completed_count + elem->pup.issued_count < elem->pup.count) {
        subreq_chunk_s *chunk;

        rc = alloc_chunk(&chunk);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (chunk == NULL) {
            break;
        }

        if (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__PACK &&
            ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU &&
            ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__REGISTERED_HOST) {
            const char *sbuf = (const char *) elem->pup.inbuf +
                chunk->count_offset * elem->pup.type->extent;
            char *dbuf = (char *) elem->pup.outbuf + chunk->count_offset * elem->pup.type->size;

            rc = yaksuri_global.gpudriver[id].info->ipack(sbuf, dbuf, chunk->count, elem->pup.type,
                                                          elem->info, &chunk->event,
                                                          chunk->gpu_tmpbuf,
                                                          ELEM_TO_INATTR_DEVICE(elem));
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__PACK &&
                   ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU &&
                   ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__UNREGISTERED_HOST) {
            const char *sbuf = (const char *) elem->pup.inbuf +
                chunk->count_offset * elem->pup.type->extent;

            rc = yaksuri_global.gpudriver[id].info->ipack(sbuf, chunk->host_tmpbuf, chunk->count,
                                                          elem->pup.type, elem->info,
                                                          &chunk->event, chunk->gpu_tmpbuf,
                                                          ELEM_TO_INATTR_DEVICE(elem));
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if ((ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__PACK &&
                    ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__REGISTERED_HOST &&
                    ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU) ||
                   (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__PACK &&
                    ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__UNREGISTERED_HOST &&
                    ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU)) {
            const char *sbuf = (const char *) elem->pup.inbuf +
                chunk->count_offset * elem->pup.type->extent;
            char *dbuf = (char *) elem->pup.outbuf + chunk->count_offset * elem->pup.type->size;

            rc = yaksuri_seq_ipack(sbuf, chunk->host_tmpbuf, chunk->count, elem->pup.type,
                                   elem->info);
            YAKSU_ERR_CHECK(rc, fn_fail);

            rc = yaksuri_global.gpudriver[id].info->ipack(chunk->host_tmpbuf, dbuf,
                                                          chunk->count * elem->pup.type->size,
                                                          byte_type, elem->info, &chunk->event,
                                                          NULL, ELEM_TO_OUTATTR_DEVICE(elem));
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__PACK &&
                   ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU &&
                   ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU) {
            assert(ELEM_TO_INATTR_DEVICE(elem) != ELEM_TO_OUTATTR_DEVICE(elem));
            const char *sbuf = (const char *) elem->pup.inbuf +
                chunk->count_offset * elem->pup.type->extent;
            char *dbuf = (char *) elem->pup.outbuf + chunk->count_offset * elem->pup.type->size;

            bool is_enabled;
            rc = yaksuri_global.gpudriver[id].info->check_p2p_comm(ELEM_TO_INATTR_DEVICE(elem),
                                                                   ELEM_TO_OUTATTR_DEVICE(elem),
                                                                   &is_enabled);
            YAKSU_ERR_CHECK(rc, fn_fail);

            if (is_enabled) {
                rc = yaksuri_global.gpudriver[id].info->ipack(sbuf, dbuf, chunk->count,
                                                              elem->pup.type, elem->info,
                                                              &chunk->event, chunk->gpu_tmpbuf,
                                                              ELEM_TO_INATTR_DEVICE(elem));
                YAKSU_ERR_CHECK(rc, fn_fail);
            } else {
                rc = yaksuri_global.gpudriver[id].info->ipack(sbuf, chunk->host_tmpbuf,
                                                              chunk->count, elem->pup.type,
                                                              elem->info, &chunk->interm_event,
                                                              chunk->gpu_tmpbuf,
                                                              ELEM_TO_INATTR_DEVICE(elem));
                YAKSU_ERR_CHECK(rc, fn_fail);

                rc = yaksuri_global.gpudriver[id].info->event_add_dependency(chunk->interm_event,
                                                                             ELEM_TO_OUTATTR_DEVICE
                                                                             (elem));
                YAKSU_ERR_CHECK(rc, fn_fail);

                rc = yaksuri_global.gpudriver[id].info->ipack(chunk->host_tmpbuf, dbuf,
                                                              chunk->count * elem->pup.type->size,
                                                              byte_type, elem->info, &chunk->event,
                                                              NULL, ELEM_TO_OUTATTR_DEVICE(elem));
                YAKSU_ERR_CHECK(rc, fn_fail);
            }
        } else if (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__UNPACK &&
                   ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__REGISTERED_HOST &&
                   ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU) {
            const char *sbuf = (const char *) elem->pup.inbuf +
                chunk->count_offset * elem->pup.type->size;
            char *dbuf = (char *) elem->pup.outbuf + chunk->count_offset * elem->pup.type->extent;

            rc = yaksuri_global.gpudriver[id].info->iunpack(sbuf, dbuf, chunk->count,
                                                            elem->pup.type, elem->info,
                                                            &chunk->event, chunk->gpu_tmpbuf,
                                                            ELEM_TO_OUTATTR_DEVICE(elem));
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__UNPACK &&
                   ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__UNREGISTERED_HOST &&
                   ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU) {
            const char *sbuf = (const char *) elem->pup.inbuf +
                chunk->count_offset * elem->pup.type->size;
            char *dbuf = (char *) elem->pup.outbuf + chunk->count_offset * elem->pup.type->extent;

            rc = yaksuri_seq_iunpack(sbuf, chunk->host_tmpbuf,
                                     chunk->count * elem->pup.type->size, byte_type, elem->info);
            YAKSU_ERR_CHECK(rc, fn_fail);

            rc = yaksuri_global.gpudriver[id].info->iunpack(chunk->host_tmpbuf, dbuf, chunk->count,
                                                            elem->pup.type, elem->info,
                                                            &chunk->event, chunk->gpu_tmpbuf,
                                                            ELEM_TO_OUTATTR_DEVICE(elem));
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if ((ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__UNPACK &&
                    ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU &&
                    ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__REGISTERED_HOST) ||
                   (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__UNPACK &&
                    ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU &&
                    ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__UNREGISTERED_HOST)) {
            const char *sbuf = (const char *) elem->pup.inbuf +
                chunk->count_offset * elem->pup.type->size;

            rc = yaksuri_global.gpudriver[id].info->iunpack(sbuf, chunk->host_tmpbuf,
                                                            chunk->count * elem->pup.type->size,
                                                            byte_type, elem->info,
                                                            &chunk->event, NULL,
                                                            ELEM_TO_INATTR_DEVICE(elem));
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if (ELEM_TO_OPTYPE(elem) == YAKSURI_OPTYPE__UNPACK &&
                   ELEM_TO_INATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU &&
                   ELEM_TO_OUTATTR_TYPE(elem) == YAKSUR_PTR_TYPE__GPU) {
            const char *sbuf = (const char *) elem->pup.inbuf +
                chunk->count_offset * elem->pup.type->size;
            char *dbuf = (char *) elem->pup.outbuf + chunk->count_offset * elem->pup.type->extent;

            bool is_enabled;
            rc = yaksuri_global.gpudriver[id].info->check_p2p_comm(ELEM_TO_INATTR_DEVICE(elem),
                                                                   ELEM_TO_OUTATTR_DEVICE(elem),
                                                                   &is_enabled);
            YAKSU_ERR_CHECK(rc, fn_fail);

            if (is_enabled) {
                rc = yaksuri_global.gpudriver[id].info->iunpack(sbuf, dbuf, chunk->count,
                                                                elem->pup.type, elem->info,
                                                                &chunk->event, chunk->gpu_tmpbuf,
                                                                ELEM_TO_INATTR_DEVICE(elem));
                YAKSU_ERR_CHECK(rc, fn_fail);
            } else {
                rc = yaksuri_global.gpudriver[id].info->iunpack(sbuf, chunk->host_tmpbuf,
                                                                chunk->count * elem->pup.type->size,
                                                                byte_type, elem->info,
                                                                &chunk->interm_event, NULL,
                                                                ELEM_TO_INATTR_DEVICE(elem));
                YAKSU_ERR_CHECK(rc, fn_fail);

                rc = yaksuri_global.gpudriver[id].info->event_add_dependency(chunk->interm_event,
                                                                             ELEM_TO_OUTATTR_DEVICE
                                                                             (elem));
                YAKSU_ERR_CHECK(rc, fn_fail);

                rc = yaksuri_global.gpudriver[id].info->iunpack(chunk->host_tmpbuf, dbuf,
                                                                chunk->count, elem->pup.type,
                                                                elem->info, &chunk->event,
                                                                chunk->gpu_tmpbuf,
                                                                ELEM_TO_OUTATTR_DEVICE(elem));
                YAKSU_ERR_CHECK(rc, fn_fail);
            }
        } else {
            rc = YAKSA_ERR__INTERNAL;
            goto fn_fail;
        }

        elem->pup.issued_count += chunk->count;
    }

  fn_exit:
    pthread_mutex_unlock(&progress_mutex);
    return rc;
  fn_fail:
    goto fn_exit;
}
