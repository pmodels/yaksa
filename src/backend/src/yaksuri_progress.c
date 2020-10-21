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

typedef struct progress_subop_s {
    uintptr_t count_offset;
    uintptr_t count;
    void *gpu_tmpbuf;
    void *host_tmpbuf;
    void *interm_event;
    void *event;

    struct progress_subop_s *next;
} progress_subop_s;

typedef struct progress_elem_s {
    struct {
        yaksuri_puptype_e puptype;

        yaksur_ptr_attr_s inattr;
        yaksur_ptr_attr_s outattr;

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
    yaksi_info_s *info;
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

    return rc;
}

int yaksuri_progress_enqueue(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                             yaksi_info_s * info, yaksi_request_s * request,
                             yaksur_ptr_attr_s inattr, yaksur_ptr_attr_s outattr,
                             yaksuri_puptype_e puptype)
{
    int rc = YAKSA_SUCCESS;

    /* if we need to go through the progress engine, make sure we only
     * take on types, where at least one count of the type fits into
     * our temporary buffers. */
    if (type->size > TMPBUF_SLAB_SIZE) {
        rc = YAKSA_ERR__NOT_SUPPORTED;
        goto fn_exit;
    }

    /* enqueue to the progress engine */
    progress_elem_s *newelem;
    newelem = (progress_elem_s *) malloc(sizeof(progress_elem_s));

    newelem->pup.puptype = puptype;
    newelem->pup.inattr = inattr;
    newelem->pup.outattr = outattr;
    newelem->pup.inbuf = inbuf;
    newelem->pup.outbuf = outbuf;
    newelem->pup.count = count;
    newelem->pup.type = type;
    newelem->pup.completed_count = 0;
    newelem->pup.issued_count = 0;
    newelem->pup.subop_head = newelem->pup.subop_tail = NULL;
    newelem->request = request;
    newelem->info = info;
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
}

static int alloc_subop(progress_subop_s ** subop)
{
    int rc = YAKSA_SUCCESS;
    progress_elem_s *elem = progress_head;
    uintptr_t gpu_tmpbuf_offset = 0, host_tmpbuf_offset = 0;
    uintptr_t nelems = UINTPTR_MAX;
    yaksuri_request_s *request_backend = (yaksuri_request_s *) elem->request->backend.priv;
    yaksuri_gpudriver_id_e id = request_backend->gpudriver_id;
    bool need_gpu_tmpbuf = false, need_host_tmpbuf = false;
    int devid = INT_MIN;

    if ((elem->pup.puptype == YAKSURI_PUPTYPE__PACK &&
         elem->pup.inattr.type == YAKSUR_PTR_TYPE__GPU &&
         elem->pup.outattr.type == YAKSUR_PTR_TYPE__UNREGISTERED_HOST) ||
        (elem->pup.puptype == YAKSURI_PUPTYPE__PACK &&
         elem->pup.inattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST &&
         elem->pup.outattr.type == YAKSUR_PTR_TYPE__GPU) ||
        (elem->pup.puptype == YAKSURI_PUPTYPE__PACK &&
         elem->pup.inattr.type == YAKSUR_PTR_TYPE__UNREGISTERED_HOST &&
         elem->pup.outattr.type == YAKSUR_PTR_TYPE__GPU) ||
        (elem->pup.puptype == YAKSURI_PUPTYPE__UNPACK &&
         elem->pup.inattr.type == YAKSUR_PTR_TYPE__UNREGISTERED_HOST &&
         elem->pup.outattr.type == YAKSUR_PTR_TYPE__GPU) ||
        (elem->pup.puptype == YAKSURI_PUPTYPE__UNPACK &&
         elem->pup.inattr.type == YAKSUR_PTR_TYPE__GPU &&
         elem->pup.outattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST) ||
        (elem->pup.puptype == YAKSURI_PUPTYPE__UNPACK &&
         elem->pup.inattr.type == YAKSUR_PTR_TYPE__GPU &&
         elem->pup.outattr.type == YAKSUR_PTR_TYPE__UNREGISTERED_HOST)) {
        need_host_tmpbuf = true;
    }

    if (elem->pup.inattr.type == YAKSUR_PTR_TYPE__GPU &&
        elem->pup.outattr.type == YAKSUR_PTR_TYPE__GPU) {
        bool is_enabled;
        rc = yaksuri_global.gpudriver[id].info->check_p2p_comm(elem->pup.inattr.device,
                                                               elem->pup.outattr.device,
                                                               &is_enabled);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (!is_enabled) {
            need_host_tmpbuf = true;
        }
    }

    if ((elem->pup.puptype == YAKSURI_PUPTYPE__PACK &&
         elem->pup.inattr.type == YAKSUR_PTR_TYPE__GPU &&
         elem->pup.outattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST) ||
        (elem->pup.puptype == YAKSURI_PUPTYPE__PACK &&
         elem->pup.inattr.type == YAKSUR_PTR_TYPE__GPU &&
         elem->pup.outattr.type == YAKSUR_PTR_TYPE__UNREGISTERED_HOST) ||
        (elem->pup.puptype == YAKSURI_PUPTYPE__PACK &&
         elem->pup.inattr.type == YAKSUR_PTR_TYPE__GPU &&
         elem->pup.outattr.type == YAKSUR_PTR_TYPE__GPU)) {
        need_gpu_tmpbuf = true;
        devid = elem->pup.inattr.device;
    }

    if ((elem->pup.puptype == YAKSURI_PUPTYPE__UNPACK &&
         elem->pup.inattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST &&
         elem->pup.outattr.type == YAKSUR_PTR_TYPE__GPU) ||
        (elem->pup.puptype == YAKSURI_PUPTYPE__UNPACK &&
         elem->pup.inattr.type == YAKSUR_PTR_TYPE__UNREGISTERED_HOST &&
         elem->pup.outattr.type == YAKSUR_PTR_TYPE__GPU) ||
        (elem->pup.puptype == YAKSURI_PUPTYPE__UNPACK &&
         elem->pup.inattr.type == YAKSUR_PTR_TYPE__GPU &&
         elem->pup.outattr.type == YAKSUR_PTR_TYPE__GPU)) {
        need_gpu_tmpbuf = true;
        devid = elem->pup.outattr.device;
    }

    assert(need_host_tmpbuf || need_gpu_tmpbuf);

    *subop = NULL;

    /* figure out if we actually have enough buffer space */
    if (need_gpu_tmpbuf) {
        uintptr_t d_nelems;
        if (yaksuri_global.gpudriver[id].device[devid].slab_head_offset == 0 &&
            yaksuri_global.gpudriver[id].device[devid].slab_tail_offset == 0) {
            d_nelems = TMPBUF_SLAB_SIZE / elem->pup.type->size;
            gpu_tmpbuf_offset = yaksuri_global.gpudriver[id].device[devid].slab_tail_offset;
        } else if (yaksuri_global.gpudriver[id].device[devid].slab_tail_offset >
                   yaksuri_global.gpudriver[id].device[devid].slab_head_offset) {
            uintptr_t count =
                (TMPBUF_SLAB_SIZE -
                 yaksuri_global.gpudriver[id].device[devid].slab_tail_offset) /
                elem->pup.type->size;
            if (count) {
                d_nelems = count;
                gpu_tmpbuf_offset = yaksuri_global.gpudriver[id].device[devid].slab_tail_offset;
            } else {
                d_nelems =
                    yaksuri_global.gpudriver[id].device[devid].slab_head_offset /
                    elem->pup.type->size;
                gpu_tmpbuf_offset = 0;
            }
        } else {
            d_nelems =
                (yaksuri_global.gpudriver[id].device[devid].slab_head_offset -
                 yaksuri_global.gpudriver[id].device[devid].slab_tail_offset) /
                elem->pup.type->size;
            gpu_tmpbuf_offset = yaksuri_global.gpudriver[id].device[devid].slab_tail_offset;
        }

        if (nelems > d_nelems)
            nelems = d_nelems;
    }

    if (need_host_tmpbuf) {
        uintptr_t h_nelems;
        if (yaksuri_global.gpudriver[id].host.slab_head_offset == 0 &&
            yaksuri_global.gpudriver[id].host.slab_tail_offset == 0) {
            h_nelems = TMPBUF_SLAB_SIZE / elem->pup.type->size;
            host_tmpbuf_offset = yaksuri_global.gpudriver[id].host.slab_tail_offset;
        } else if (yaksuri_global.gpudriver[id].host.slab_tail_offset >
                   yaksuri_global.gpudriver[id].host.slab_head_offset) {
            uintptr_t count =
                (TMPBUF_SLAB_SIZE -
                 yaksuri_global.gpudriver[id].host.slab_tail_offset) / elem->pup.type->size;
            if (count) {
                h_nelems = count;
                host_tmpbuf_offset = yaksuri_global.gpudriver[id].host.slab_tail_offset;
            } else {
                h_nelems =
                    yaksuri_global.gpudriver[id].host.slab_head_offset / elem->pup.type->size;
                host_tmpbuf_offset = 0;
            }
        } else {
            h_nelems =
                (yaksuri_global.gpudriver[id].host.slab_head_offset -
                 yaksuri_global.gpudriver[id].host.slab_tail_offset) / elem->pup.type->size;
            host_tmpbuf_offset = yaksuri_global.gpudriver[id].host.slab_tail_offset;
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
    if (need_gpu_tmpbuf) {
        if (yaksuri_global.gpudriver[id].device[devid].slab == NULL) {
            yaksuri_global.gpudriver[id].device[devid].slab =
                yaksuri_global.gpudriver[id].info->gpu_malloc(TMPBUF_SLAB_SIZE, id);
        }
        yaksuri_global.gpudriver[id].device[devid].slab_tail_offset =
            gpu_tmpbuf_offset + nelems * elem->pup.type->size;
    }

    if (need_host_tmpbuf) {
        if (yaksuri_global.gpudriver[id].host.slab == NULL) {
            yaksuri_global.gpudriver[id].host.slab =
                yaksuri_global.gpudriver[id].info->host_malloc(TMPBUF_SLAB_SIZE);
        }
        yaksuri_global.gpudriver[id].host.slab_tail_offset =
            host_tmpbuf_offset + nelems * elem->pup.type->size;
    }


    /* allocate the subop */
    *subop = (progress_subop_s *) malloc(sizeof(progress_subop_s));

    (*subop)->count_offset = elem->pup.completed_count + elem->pup.issued_count;
    (*subop)->count = nelems;
    if (need_gpu_tmpbuf)
        (*subop)->gpu_tmpbuf =
            (void *) ((char *) yaksuri_global.gpudriver[id].device[devid].slab + gpu_tmpbuf_offset);
    else
        (*subop)->gpu_tmpbuf = NULL;

    if (need_host_tmpbuf)
        (*subop)->host_tmpbuf =
            (void *) ((char *) yaksuri_global.gpudriver[id].host.slab + host_tmpbuf_offset);
    else
        (*subop)->host_tmpbuf = NULL;

    (*subop)->interm_event = NULL;
    (*subop)->event = NULL;
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
    yaksuri_gpudriver_id_e id = request_backend->gpudriver_id;

    if (subop->interm_event) {
        rc = yaksuri_global.gpudriver[id].info->event_destroy(subop->interm_event);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    rc = yaksuri_global.gpudriver[id].info->event_destroy(subop->event);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* free the device buffer */
    if (subop->gpu_tmpbuf) {
        int devid;

        if (elem->pup.puptype == YAKSURI_PUPTYPE__PACK)
            devid = elem->pup.inattr.device;
        else
            devid = elem->pup.outattr.device;

        assert(subop->gpu_tmpbuf ==
               (char *) yaksuri_global.gpudriver[id].device[devid].slab +
               yaksuri_global.gpudriver[id].device[devid].slab_head_offset);
        if (subop->next) {
            yaksuri_global.gpudriver[id].device[devid].slab_head_offset =
                (uintptr_t) ((char *) subop->next->gpu_tmpbuf -
                             (char *) yaksuri_global.gpudriver[id].device[devid].slab);
        } else {
            yaksuri_global.gpudriver[id].device[devid].slab_head_offset =
                yaksuri_global.gpudriver[id].device[devid].slab_tail_offset = 0;
        }
    }

    /* free the host buffer */
    if (subop->host_tmpbuf) {
        assert(subop->host_tmpbuf ==
               (char *) yaksuri_global.gpudriver[id].host.slab +
               yaksuri_global.gpudriver[id].host.slab_head_offset);
        if (subop->next) {
            yaksuri_global.gpudriver[id].host.slab_head_offset =
                (uintptr_t) ((char *) subop->next->gpu_tmpbuf -
                             (char *) yaksuri_global.gpudriver[id].host.slab);
        } else {
            yaksuri_global.gpudriver[id].host.slab_head_offset =
                yaksuri_global.gpudriver[id].host.slab_tail_offset = 0;
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
                break;
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

    progress_elem_s *elem;
    elem = progress_head;
    yaksuri_request_s *request_backend;
    request_backend = (yaksuri_request_s *) elem->request->backend.priv;
    yaksuri_gpudriver_id_e id;
    id = request_backend->gpudriver_id;

    /****************************************************************************/
    /* Step 1: Check for completion and free up any held up resources */
    /****************************************************************************/
    for (progress_subop_s * subop = elem->pup.subop_head; subop;) {
        int completed;
        rc = yaksuri_global.gpudriver[id].info->event_query(subop->event, &completed);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (completed) {
            if (elem->pup.puptype == YAKSURI_PUPTYPE__PACK &&
                elem->pup.inattr.type == YAKSUR_PTR_TYPE__GPU &&
                elem->pup.outattr.type == YAKSUR_PTR_TYPE__UNREGISTERED_HOST) {
                char *dbuf = (char *) elem->pup.outbuf + subop->count_offset * elem->pup.type->size;
                rc = yaksuri_seq_ipack(subop->host_tmpbuf, dbuf,
                                       subop->count * elem->pup.type->size, byte_type, elem->info);
                YAKSU_ERR_CHECK(rc, fn_fail);
            } else if (elem->pup.puptype == YAKSURI_PUPTYPE__UNPACK &&
                       elem->pup.inattr.type == YAKSUR_PTR_TYPE__GPU &&
                       (elem->pup.outattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST ||
                        elem->pup.outattr.type == YAKSUR_PTR_TYPE__UNREGISTERED_HOST)) {
                char *dbuf =
                    (char *) elem->pup.outbuf + subop->count_offset * elem->pup.type->extent;
                rc = yaksuri_seq_iunpack(subop->host_tmpbuf, dbuf, subop->count, elem->pup.type,
                                         elem->info);
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

        rc = alloc_subop(&subop);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (subop == NULL) {
            break;
        }

        if (elem->pup.puptype == YAKSURI_PUPTYPE__PACK &&
            elem->pup.inattr.type == YAKSUR_PTR_TYPE__GPU &&
            elem->pup.outattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST) {
            const char *sbuf = (const char *) elem->pup.inbuf +
                subop->count_offset * elem->pup.type->extent;
            char *dbuf = (char *) elem->pup.outbuf + subop->count_offset * elem->pup.type->size;

            rc = yaksuri_global.gpudriver[id].info->ipack(sbuf, dbuf, subop->count, elem->pup.type,
                                                          elem->info, &subop->event,
                                                          subop->gpu_tmpbuf,
                                                          elem->pup.inattr.device);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if (elem->pup.puptype == YAKSURI_PUPTYPE__PACK &&
                   elem->pup.inattr.type == YAKSUR_PTR_TYPE__GPU &&
                   elem->pup.outattr.type == YAKSUR_PTR_TYPE__UNREGISTERED_HOST) {
            const char *sbuf = (const char *) elem->pup.inbuf +
                subop->count_offset * elem->pup.type->extent;

            rc = yaksuri_global.gpudriver[id].info->ipack(sbuf, subop->host_tmpbuf, subop->count,
                                                          elem->pup.type, elem->info,
                                                          &subop->event, subop->gpu_tmpbuf,
                                                          elem->pup.inattr.device);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if ((elem->pup.puptype == YAKSURI_PUPTYPE__PACK &&
                    elem->pup.inattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST &&
                    elem->pup.outattr.type == YAKSUR_PTR_TYPE__GPU) ||
                   (elem->pup.puptype == YAKSURI_PUPTYPE__PACK &&
                    elem->pup.inattr.type == YAKSUR_PTR_TYPE__UNREGISTERED_HOST &&
                    elem->pup.outattr.type == YAKSUR_PTR_TYPE__GPU)) {
            const char *sbuf = (const char *) elem->pup.inbuf +
                subop->count_offset * elem->pup.type->extent;
            char *dbuf = (char *) elem->pup.outbuf + subop->count_offset * elem->pup.type->size;

            rc = yaksuri_seq_ipack(sbuf, subop->host_tmpbuf, subop->count, elem->pup.type,
                                   elem->info);
            YAKSU_ERR_CHECK(rc, fn_fail);

            rc = yaksuri_global.gpudriver[id].info->ipack(subop->host_tmpbuf, dbuf,
                                                          subop->count * elem->pup.type->size,
                                                          byte_type, elem->info, &subop->event,
                                                          NULL, elem->pup.outattr.device);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if (elem->pup.puptype == YAKSURI_PUPTYPE__PACK &&
                   elem->pup.inattr.type == YAKSUR_PTR_TYPE__GPU &&
                   elem->pup.outattr.type == YAKSUR_PTR_TYPE__GPU) {
            assert(elem->pup.inattr.device != elem->pup.outattr.device);
            const char *sbuf = (const char *) elem->pup.inbuf +
                subop->count_offset * elem->pup.type->extent;
            char *dbuf = (char *) elem->pup.outbuf + subop->count_offset * elem->pup.type->size;

            bool is_enabled;
            rc = yaksuri_global.gpudriver[id].info->check_p2p_comm(elem->pup.inattr.device,
                                                                   elem->pup.outattr.device,
                                                                   &is_enabled);
            YAKSU_ERR_CHECK(rc, fn_fail);

            if (is_enabled) {
                rc = yaksuri_global.gpudriver[id].info->ipack(sbuf, dbuf, subop->count,
                                                              elem->pup.type, elem->info,
                                                              &subop->event, subop->gpu_tmpbuf,
                                                              elem->pup.inattr.device);
                YAKSU_ERR_CHECK(rc, fn_fail);
            } else {
                rc = yaksuri_global.gpudriver[id].info->ipack(sbuf, subop->host_tmpbuf,
                                                              subop->count, elem->pup.type,
                                                              elem->info, &subop->interm_event,
                                                              subop->gpu_tmpbuf,
                                                              elem->pup.inattr.device);
                YAKSU_ERR_CHECK(rc, fn_fail);

                rc = yaksuri_global.gpudriver[id].info->event_add_dependency(subop->interm_event,
                                                                             elem->pup.
                                                                             outattr.device);
                YAKSU_ERR_CHECK(rc, fn_fail);

                rc = yaksuri_global.gpudriver[id].info->ipack(subop->host_tmpbuf, dbuf,
                                                              subop->count * elem->pup.type->size,
                                                              byte_type, elem->info, &subop->event,
                                                              NULL, elem->pup.outattr.device);
                YAKSU_ERR_CHECK(rc, fn_fail);
            }
        } else if (elem->pup.puptype == YAKSURI_PUPTYPE__UNPACK &&
                   elem->pup.inattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST &&
                   elem->pup.outattr.type == YAKSUR_PTR_TYPE__GPU) {
            const char *sbuf = (const char *) elem->pup.inbuf +
                subop->count_offset * elem->pup.type->size;
            char *dbuf = (char *) elem->pup.outbuf + subop->count_offset * elem->pup.type->extent;

            rc = yaksuri_global.gpudriver[id].info->iunpack(sbuf, dbuf, subop->count,
                                                            elem->pup.type, elem->info,
                                                            &subop->event, subop->gpu_tmpbuf,
                                                            elem->pup.outattr.device);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if (elem->pup.puptype == YAKSURI_PUPTYPE__UNPACK &&
                   elem->pup.inattr.type == YAKSUR_PTR_TYPE__UNREGISTERED_HOST &&
                   elem->pup.outattr.type == YAKSUR_PTR_TYPE__GPU) {
            const char *sbuf = (const char *) elem->pup.inbuf +
                subop->count_offset * elem->pup.type->size;
            char *dbuf = (char *) elem->pup.outbuf + subop->count_offset * elem->pup.type->extent;

            rc = yaksuri_seq_iunpack(sbuf, subop->host_tmpbuf,
                                     subop->count * elem->pup.type->size, byte_type, elem->info);
            YAKSU_ERR_CHECK(rc, fn_fail);

            rc = yaksuri_global.gpudriver[id].info->iunpack(subop->host_tmpbuf, dbuf, subop->count,
                                                            elem->pup.type, elem->info,
                                                            &subop->event, subop->gpu_tmpbuf,
                                                            elem->pup.outattr.device);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if ((elem->pup.puptype == YAKSURI_PUPTYPE__UNPACK &&
                    elem->pup.inattr.type == YAKSUR_PTR_TYPE__GPU &&
                    elem->pup.outattr.type == YAKSUR_PTR_TYPE__REGISTERED_HOST) ||
                   (elem->pup.puptype == YAKSURI_PUPTYPE__UNPACK &&
                    elem->pup.inattr.type == YAKSUR_PTR_TYPE__GPU &&
                    elem->pup.outattr.type == YAKSUR_PTR_TYPE__UNREGISTERED_HOST)) {
            const char *sbuf = (const char *) elem->pup.inbuf +
                subop->count_offset * elem->pup.type->size;

            rc = yaksuri_global.gpudriver[id].info->iunpack(sbuf, subop->host_tmpbuf,
                                                            subop->count * elem->pup.type->size,
                                                            byte_type, elem->info,
                                                            &subop->event, NULL,
                                                            elem->pup.inattr.device);
            YAKSU_ERR_CHECK(rc, fn_fail);
        } else if (elem->pup.puptype == YAKSURI_PUPTYPE__UNPACK &&
                   elem->pup.inattr.type == YAKSUR_PTR_TYPE__GPU &&
                   elem->pup.outattr.type == YAKSUR_PTR_TYPE__GPU) {
            const char *sbuf = (const char *) elem->pup.inbuf +
                subop->count_offset * elem->pup.type->size;
            char *dbuf = (char *) elem->pup.outbuf + subop->count_offset * elem->pup.type->extent;

            bool is_enabled;
            rc = yaksuri_global.gpudriver[id].info->check_p2p_comm(elem->pup.inattr.device,
                                                                   elem->pup.outattr.device,
                                                                   &is_enabled);
            YAKSU_ERR_CHECK(rc, fn_fail);

            if (is_enabled) {
                rc = yaksuri_global.gpudriver[id].info->iunpack(sbuf, dbuf, subop->count,
                                                                elem->pup.type, elem->info,
                                                                &subop->event, subop->gpu_tmpbuf,
                                                                elem->pup.inattr.device);
                YAKSU_ERR_CHECK(rc, fn_fail);
            } else {
                rc = yaksuri_global.gpudriver[id].info->iunpack(sbuf, subop->host_tmpbuf,
                                                                subop->count * elem->pup.type->size,
                                                                byte_type, elem->info,
                                                                &subop->interm_event, NULL,
                                                                elem->pup.inattr.device);
                YAKSU_ERR_CHECK(rc, fn_fail);

                rc = yaksuri_global.gpudriver[id].info->event_add_dependency(subop->interm_event,
                                                                             elem->pup.
                                                                             outattr.device);
                YAKSU_ERR_CHECK(rc, fn_fail);

                rc = yaksuri_global.gpudriver[id].info->iunpack(subop->host_tmpbuf, dbuf,
                                                                subop->count, elem->pup.type,
                                                                elem->info, &subop->event,
                                                                subop->gpu_tmpbuf,
                                                                elem->pup.outattr.device);
                YAKSU_ERR_CHECK(rc, fn_fail);
            }
        } else {
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
