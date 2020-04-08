/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include "yaksa.h"
#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri.h"

typedef struct progress_subop_s {
    uintptr_t count_offset;
    uintptr_t count;
    void *device_tmpbuf;
    void *host_tmpbuf;
    void *event;

    struct progress_subop_s *next;
} progress_subop_s;

typedef struct progress_elem_s {
    yaksuri_progress_elem_kind_e kind;

    struct {
        const void *inbuf;
        void *outbuf;
        uintptr_t count;
        yaksi_type_s *type;

        uintptr_t completed_count;
        uintptr_t issued_count;
        progress_subop_s *subop_head;
        progress_subop_s *subop_tail;
    } pup;

    yaksi_request_s *request;
    struct progress_elem_s *next;
} progress_elem_s;

static progress_elem_s *progress_head = NULL;
static progress_elem_s *progress_tail = NULL;
static pthread_mutex_t progress_mutex = PTHREAD_MUTEX_INITIALIZER;

#define TMPBUF_SLAB_SIZE  (16 * 1024 * 1024)

/* the dequeue function is not thread safe, as it is always called
 * from within the progress engine */
static int progress_dequeue(progress_elem_s * elem)
{
    int rc = YAKSA_SUCCESS;

    assert(progress_head);

    if (progress_head == elem && progress_tail == elem) {
        progress_head = progress_tail = NULL;
    } else if (progress_head == elem) {
        progress_head = progress_head->next;
    } else {
        progress_elem_s *tmp;
        for (tmp = progress_head; tmp->next; tmp = tmp->next)
            if (tmp->next == elem)
                break;
        assert(tmp->next);
        tmp->next = tmp->next->next;
        if (tmp->next == NULL)
            progress_tail = tmp;
    }

    yaksu_atomic_decr(&elem->request->cc);
    free(elem);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_progress_enqueue(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                             yaksi_request_s * request, yaksuri_progress_elem_kind_e kind)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_type_s *type_backend = (yaksuri_type_s *) type->backend.priv;
    yaksuri_request_s *request_backend = (yaksuri_request_s *) request->backend.priv;
    yaksuri_gpudev_id_e id = request_backend->gpudev_id;

    /* if we need to go through the progress engine, make sure we only
     * take on types, where at least one count of the type fits into
     * our temporary buffers. */
    if (type->size > TMPBUF_SLAB_SIZE || type_backend->gpudev[id].pack == NULL) {
        rc = YAKSA_ERR__NOT_SUPPORTED;
        goto fn_exit;
    }

    /* enqueue to the progress engine */
    progress_elem_s *newelem;
    newelem = (progress_elem_s *) malloc(sizeof(progress_elem_s));

    newelem->kind = kind;
    newelem->pup.inbuf = inbuf;
    newelem->pup.outbuf = outbuf;
    newelem->pup.count = count;
    newelem->pup.type = type;
    newelem->pup.completed_count = 0;
    newelem->pup.issued_count = 0;
    newelem->pup.subop_head = newelem->pup.subop_tail = NULL;
    newelem->request = request;
    newelem->next = NULL;

    /* enqueue the new element */
    yaksu_atomic_incr(&request->cc);
    pthread_mutex_lock(&progress_mutex);
    if (progress_tail == NULL) {
        progress_head = progress_tail = newelem;
    } else {
        progress_tail->next = newelem;
        progress_tail = newelem;
    }
    pthread_mutex_unlock(&progress_mutex);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int alloc_subop(yaksuri_gpudev_id_e id, bool need_host_buf, bool need_device_buf,
                       progress_subop_s ** subop)
{
    int rc = YAKSA_SUCCESS;
    progress_elem_s *elem = progress_head;
    uintptr_t device_tmpbuf_offset = 0, host_tmpbuf_offset = 0;
    uintptr_t nelems = UINTPTR_MAX;

    assert(need_host_buf || need_device_buf);

    *subop = NULL;

    /* figure out if we actually have enough buffer space */
    if (need_device_buf) {
        uintptr_t d_nelems;
        if (yaksuri_global.gpudev[id].device.slab_head_offset == 0 &&
            yaksuri_global.gpudev[id].device.slab_tail_offset == 0) {
            d_nelems = TMPBUF_SLAB_SIZE / elem->pup.type->size;
            device_tmpbuf_offset = yaksuri_global.gpudev[id].device.slab_tail_offset;
        } else if (yaksuri_global.gpudev[id].device.slab_tail_offset >
                   yaksuri_global.gpudev[id].device.slab_head_offset) {
            uintptr_t count =
                (TMPBUF_SLAB_SIZE -
                 yaksuri_global.gpudev[id].device.slab_tail_offset) / elem->pup.type->size;
            if (count) {
                d_nelems = count;
                device_tmpbuf_offset = yaksuri_global.gpudev[id].device.slab_tail_offset;
            } else {
                d_nelems = yaksuri_global.gpudev[id].device.slab_head_offset / elem->pup.type->size;
                device_tmpbuf_offset = 0;
            }
        } else {
            d_nelems =
                (yaksuri_global.gpudev[id].device.slab_head_offset -
                 yaksuri_global.gpudev[id].device.slab_tail_offset) / elem->pup.type->size;
            device_tmpbuf_offset = yaksuri_global.gpudev[id].device.slab_tail_offset;
        }

        if (nelems > d_nelems)
            nelems = d_nelems;
    }

    if (need_host_buf) {
        uintptr_t h_nelems;
        if (yaksuri_global.gpudev[id].host.slab_head_offset == 0 &&
            yaksuri_global.gpudev[id].host.slab_tail_offset == 0) {
            h_nelems = TMPBUF_SLAB_SIZE / elem->pup.type->size;
            host_tmpbuf_offset = yaksuri_global.gpudev[id].host.slab_tail_offset;
        } else if (yaksuri_global.gpudev[id].host.slab_tail_offset >
                   yaksuri_global.gpudev[id].host.slab_head_offset) {
            uintptr_t count =
                (TMPBUF_SLAB_SIZE -
                 yaksuri_global.gpudev[id].host.slab_tail_offset) / elem->pup.type->size;
            if (count) {
                h_nelems = count;
                host_tmpbuf_offset = yaksuri_global.gpudev[id].host.slab_tail_offset;
            } else {
                h_nelems = yaksuri_global.gpudev[id].host.slab_head_offset / elem->pup.type->size;
                host_tmpbuf_offset = 0;
            }
        } else {
            h_nelems =
                (yaksuri_global.gpudev[id].host.slab_head_offset -
                 yaksuri_global.gpudev[id].host.slab_tail_offset) / elem->pup.type->size;
            host_tmpbuf_offset = yaksuri_global.gpudev[id].host.slab_tail_offset;
        }

        if (nelems > h_nelems)
            nelems = h_nelems;
    }

    if (nelems > elem->pup.count - elem->pup.completed_count - elem->pup.issued_count)
        nelems = elem->pup.count - elem->pup.completed_count - elem->pup.issued_count;

    /* if we don't have enough space, return */
    if (nelems == 0) {
        goto fn_exit;
    }


    /* allocate the actual buffer space */
    if (need_device_buf) {
        if (yaksuri_global.gpudev[id].device.slab == NULL) {
            yaksuri_global.gpudev[id].device.slab =
                yaksuri_global.gpudev[id].info->device_malloc(TMPBUF_SLAB_SIZE);
        }
        yaksuri_global.gpudev[id].device.slab_tail_offset =
            device_tmpbuf_offset + nelems * elem->pup.type->size;
    }

    if (need_host_buf) {
        if (yaksuri_global.gpudev[id].host.slab == NULL) {
            yaksuri_global.gpudev[id].host.slab =
                yaksuri_global.gpudev[id].info->host_malloc(TMPBUF_SLAB_SIZE);
        }
        yaksuri_global.gpudev[id].host.slab_tail_offset =
            host_tmpbuf_offset + nelems * elem->pup.type->size;
    }


    /* allocate the subop */
    *subop = (progress_subop_s *) malloc(sizeof(progress_subop_s));

    (*subop)->count_offset = elem->pup.completed_count + elem->pup.issued_count;
    (*subop)->count = nelems;
    (*subop)->device_tmpbuf =
        (void *) ((char *) yaksuri_global.gpudev[id].device.slab + device_tmpbuf_offset);
    (*subop)->host_tmpbuf =
        (void *) ((char *) yaksuri_global.gpudev[id].host.slab + host_tmpbuf_offset);

    rc = yaksuri_global.gpudev[id].info->event_create(&(*subop)->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

    (*subop)->next = NULL;

    if (elem->pup.subop_tail == NULL) {
        assert(elem->pup.subop_head == NULL);
        elem->pup.subop_head = elem->pup.subop_tail = *subop;
    } else {
        elem->pup.subop_tail->next = *subop;
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

static int free_subop(progress_subop_s * subop)
{
    int rc = YAKSA_SUCCESS;
    progress_elem_s *elem = progress_head;
    yaksuri_request_s *request_backend = (yaksuri_request_s *) elem->request->backend.priv;
    yaksuri_gpudev_id_e id = request_backend->gpudev_id;

    rc = yaksuri_global.gpudev[id].info->event_destroy(subop->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* free the device buffer */
    if (subop->device_tmpbuf) {
        assert(subop->device_tmpbuf ==
               (char *) yaksuri_global.gpudev[id].device.slab +
               yaksuri_global.gpudev[id].device.slab_head_offset);
        progress_subop_s *tmp;
        for (tmp = subop; tmp->next; tmp = tmp->next) {
            if (tmp->next->device_tmpbuf) {
                yaksuri_global.gpudev[id].device.slab_head_offset =
                    (uintptr_t) ((char *) tmp->next->device_tmpbuf -
                                 (char *) yaksuri_global.gpudev[id].device.slab);
                break;
            }
        }
        if (tmp->next == NULL) {
            yaksuri_global.gpudev[id].device.slab_head_offset =
                yaksuri_global.gpudev[id].device.slab_tail_offset = 0;
        }
    }

    /* free the host buffer */
    if (subop->host_tmpbuf) {
        assert(subop->host_tmpbuf ==
               (char *) yaksuri_global.gpudev[id].host.slab +
               yaksuri_global.gpudev[id].host.slab_head_offset);
        progress_subop_s *tmp;
        for (tmp = subop; tmp->next; tmp = tmp->next) {
            if (tmp->next->host_tmpbuf) {
                yaksuri_global.gpudev[id].host.slab_head_offset =
                    (uintptr_t) ((char *) tmp->next->host_tmpbuf -
                                 (char *) yaksuri_global.gpudev[id].host.slab);
                break;
            }
        }
        if (tmp->next == NULL) {
            yaksuri_global.gpudev[id].host.slab_head_offset =
                yaksuri_global.gpudev[id].host.slab_tail_offset = 0;
        }
    }

    if (elem->pup.subop_head == subop && elem->pup.subop_tail == subop) {
        elem->pup.subop_head = elem->pup.subop_tail = NULL;
    } else if (elem->pup.subop_head == subop) {
        elem->pup.subop_head = subop->next;
    } else {
        for (progress_subop_s * tmp = elem->pup.subop_head; tmp->next; tmp = tmp->next) {
            if (tmp->next == subop) {
                tmp->next = subop->next;
                if (elem->pup.subop_tail == subop)
                    elem->pup.subop_tail = tmp;
            }
        }
    }

    free(subop);

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
    if (progress_head == NULL)
        goto fn_exit;

    /* the progress engine has three steps: (1) check for completions
     * and free up any held up resources; (2) if we don't have
     * anything else to do, return; and (3) issue any pending
     * subops. */
    yaksi_type_s *byte_type;
    rc = yaksi_type_get(YAKSA_TYPE__BYTE, &byte_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    progress_elem_s *elem = progress_head;
    yaksuri_request_s *request_backend = (yaksuri_request_s *) elem->request->backend.priv;
    yaksuri_gpudev_id_e id = request_backend->gpudev_id;
    yaksuri_type_s *type_backend = (yaksuri_type_s *) elem->pup.type->backend.priv;
    yaksuri_type_s *byte_type_backend = (yaksuri_type_s *) byte_type->backend.priv;

    /****************************************************************************/
    /* Step 1: Check for completion and free up any held up resources */
    /****************************************************************************/
    for (progress_subop_s * subop = elem->pup.subop_head; subop;) {
        int completed;
        rc = yaksuri_global.gpudev[id].info->event_query(subop->event, &completed);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (completed) {
            if (elem->kind == YAKSURI_PROGRESS_ELEM_KIND__PACK_D2URH) {
                char *dbuf = (char *) elem->pup.outbuf + subop->count_offset * elem->pup.type->size;
                rc = byte_type_backend->seq.pack(subop->host_tmpbuf, dbuf,
                                                 subop->count * elem->pup.type->size, byte_type);
                YAKSU_ERR_CHECK(rc, fn_fail);
            } else if (elem->kind == YAKSURI_PROGRESS_ELEM_KIND__UNPACK_D2RH ||
                       elem->kind == YAKSURI_PROGRESS_ELEM_KIND__UNPACK_D2URH) {
                char *dbuf =
                    (char *) elem->pup.outbuf + subop->count_offset * elem->pup.type->extent;
                rc = type_backend->seq.unpack(subop->host_tmpbuf, dbuf, subop->count,
                                              elem->pup.type);
                YAKSU_ERR_CHECK(rc, fn_fail);
            }

            elem->pup.completed_count += subop->count;
            elem->pup.issued_count -= subop->count;

            progress_subop_s *tmp = subop->next;
            rc = free_subop(subop);
            YAKSU_ERR_CHECK(rc, fn_fail);
            subop = tmp;
        } else {
            break;
        }
    }


    /****************************************************************************/
    /* Step 2: If we don't have any more work to do, return */
    /****************************************************************************/
    if (elem->pup.completed_count == elem->pup.count) {
        rc = progress_dequeue(elem);
        YAKSU_ERR_CHECK(rc, fn_fail);
        goto fn_exit;
    }


    /****************************************************************************/
    /* Step 3: Issue any pending subops */
    /****************************************************************************/
    while (elem->pup.completed_count + elem->pup.issued_count < elem->pup.count) {
        progress_subop_s *subop;
        bool need_device_tmpbuf = false, need_host_tmpbuf = false;

        if (elem->kind == YAKSURI_PROGRESS_ELEM_KIND__PACK_D2URH ||
            elem->kind == YAKSURI_PROGRESS_ELEM_KIND__PACK_RH2D ||
            elem->kind == YAKSURI_PROGRESS_ELEM_KIND__PACK_URH2D ||
            elem->kind == YAKSURI_PROGRESS_ELEM_KIND__UNPACK_URH2D ||
            elem->kind == YAKSURI_PROGRESS_ELEM_KIND__UNPACK_D2RH ||
            elem->kind == YAKSURI_PROGRESS_ELEM_KIND__UNPACK_D2URH) {
            need_host_tmpbuf = true;
        }

        if (elem->kind == YAKSURI_PROGRESS_ELEM_KIND__PACK_D2RH ||
            elem->kind == YAKSURI_PROGRESS_ELEM_KIND__PACK_D2URH ||
            elem->kind == YAKSURI_PROGRESS_ELEM_KIND__UNPACK_RH2D ||
            elem->kind == YAKSURI_PROGRESS_ELEM_KIND__UNPACK_URH2D) {
            need_device_tmpbuf = true;
        }

        rc = alloc_subop(id, need_host_tmpbuf, need_device_tmpbuf, &subop);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (subop == NULL) {
            break;
        }

        switch (elem->kind) {
            case YAKSURI_PROGRESS_ELEM_KIND__PACK_D2RH:
                {
                    const char *sbuf = (const char *) elem->pup.inbuf +
                        subop->count_offset * elem->pup.type->extent;
                    char *dbuf =
                        (char *) elem->pup.outbuf + subop->count_offset * elem->pup.type->size;

                    rc = type_backend->gpudev[id].pack(sbuf, dbuf, subop->count, elem->pup.type,
                                                       subop->device_tmpbuf, subop->event);
                    YAKSU_ERR_CHECK(rc, fn_fail);
                }
                break;

            case YAKSURI_PROGRESS_ELEM_KIND__PACK_D2URH:
                {
                    const char *sbuf = (const char *) elem->pup.inbuf +
                        subop->count_offset * elem->pup.type->extent;

                    rc = type_backend->gpudev[id].pack(sbuf, subop->host_tmpbuf, subop->count,
                                                       elem->pup.type, subop->device_tmpbuf,
                                                       subop->event);
                    YAKSU_ERR_CHECK(rc, fn_fail);
                }
                break;

            case YAKSURI_PROGRESS_ELEM_KIND__PACK_RH2D:
            case YAKSURI_PROGRESS_ELEM_KIND__PACK_URH2D:
                {
                    const char *sbuf = (const char *) elem->pup.inbuf +
                        subop->count_offset * elem->pup.type->extent;
                    char *dbuf =
                        (char *) elem->pup.outbuf + subop->count_offset * elem->pup.type->size;

                    rc = type_backend->seq.pack(sbuf, subop->host_tmpbuf, subop->count,
                                                elem->pup.type);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    rc = byte_type_backend->gpudev[id].pack(subop->host_tmpbuf, dbuf,
                                                            subop->count * elem->pup.type->size,
                                                            byte_type, NULL, subop->event);
                    YAKSU_ERR_CHECK(rc, fn_fail);
                }
                break;

            case YAKSURI_PROGRESS_ELEM_KIND__UNPACK_RH2D:
                {
                    const char *sbuf = (const char *) elem->pup.inbuf +
                        subop->count_offset * elem->pup.type->size;
                    char *dbuf =
                        (char *) elem->pup.outbuf + subop->count_offset * elem->pup.type->extent;

                    rc = type_backend->gpudev[id].unpack(sbuf, dbuf, subop->count, elem->pup.type,
                                                         subop->device_tmpbuf, subop->event);
                    YAKSU_ERR_CHECK(rc, fn_fail);
                }
                break;

            case YAKSURI_PROGRESS_ELEM_KIND__UNPACK_URH2D:
                {
                    const char *sbuf = (const char *) elem->pup.inbuf +
                        subop->count_offset * elem->pup.type->size;
                    char *dbuf =
                        (char *) elem->pup.outbuf + subop->count_offset * elem->pup.type->extent;

                    rc = byte_type_backend->seq.unpack(sbuf, subop->host_tmpbuf,
                                                       subop->count * elem->pup.type->size,
                                                       byte_type);
                    YAKSU_ERR_CHECK(rc, fn_fail);

                    rc = type_backend->gpudev[id].unpack(subop->host_tmpbuf, dbuf, subop->count,
                                                         elem->pup.type, subop->device_tmpbuf,
                                                         subop->event);
                    YAKSU_ERR_CHECK(rc, fn_fail);
                }
                break;

            case YAKSURI_PROGRESS_ELEM_KIND__UNPACK_D2RH:
            case YAKSURI_PROGRESS_ELEM_KIND__UNPACK_D2URH:
                {
                    const char *sbuf = (const char *) elem->pup.inbuf +
                        subop->count_offset * elem->pup.type->size;

                    rc = byte_type_backend->gpudev[id].unpack(sbuf, subop->host_tmpbuf,
                                                              subop->count * elem->pup.type->size,
                                                              byte_type, NULL, subop->event);
                    YAKSU_ERR_CHECK(rc, fn_fail);
                }
                break;

            default:
                rc = YAKSA_ERR__INTERNAL;
                goto fn_fail;
        }

        elem->pup.issued_count += subop->count;
    }

  fn_exit:
    pthread_mutex_unlock(&progress_mutex);
    return rc;
  fn_fail:
    goto fn_exit;
}
