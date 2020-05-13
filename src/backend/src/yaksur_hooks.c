/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <stdlib.h>
#include <assert.h>
#include "yaksa.h"
#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri.h"

yaksuri_global_s yaksuri_global;

int yaksur_init_hook(void)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_gpudriver_id_e id;

    rc = yaksuri_seq_init_hook();
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* CUDA hooks */
    id = YAKSURI_GPUDRIVER_ID__CUDA;
    yaksuri_global.gpudriver[id].info = NULL;
    rc = yaksuri_cuda_init_hook(&yaksuri_global.gpudriver[id].info);
    YAKSU_ERR_CHECK(rc, fn_fail);


    /* final setup for all backends */
    for (id = YAKSURI_GPUDRIVER_ID__UNSET; id < YAKSURI_GPUDRIVER_ID__LAST; id++) {
        if (id == YAKSURI_GPUDRIVER_ID__UNSET)
            continue;

        if (yaksuri_global.gpudriver[id].info) {
            yaksuri_global.gpudriver[id].host.slab = NULL;
            yaksuri_global.gpudriver[id].host.slab_head_offset = 0;
            yaksuri_global.gpudriver[id].host.slab_tail_offset = 0;

            int ndevices;
            rc = yaksuri_global.gpudriver[id].info->get_num_devices(&ndevices);
            YAKSU_ERR_CHECK(rc, fn_fail);

            yaksuri_global.gpudriver[id].device = (yaksuri_slab_s *)
                malloc(ndevices * sizeof(yaksuri_slab_s));
            for (int i = 0; i < ndevices; i++) {
                yaksuri_global.gpudriver[id].device[i].slab = NULL;
                yaksuri_global.gpudriver[id].device[i].slab_head_offset = 0;
                yaksuri_global.gpudriver[id].device[i].slab_tail_offset = 0;
            }
        }
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_finalize_hook(void)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_seq_finalize_hook();
    YAKSU_ERR_CHECK(rc, fn_fail);

    for (yaksuri_gpudriver_id_e id = YAKSURI_GPUDRIVER_ID__UNSET;
         id < YAKSURI_GPUDRIVER_ID__LAST; id++) {
        if (id == YAKSURI_GPUDRIVER_ID__UNSET)
            continue;

        if (yaksuri_global.gpudriver[id].info) {
            if (yaksuri_global.gpudriver[id].host.slab) {
                yaksuri_global.gpudriver[id].info->host_free(yaksuri_global.gpudriver[id].
                                                             host.slab);
            }

            int ndevices;
            rc = yaksuri_global.gpudriver[id].info->get_num_devices(&ndevices);
            YAKSU_ERR_CHECK(rc, fn_fail);

            for (int i = 0; i < ndevices; i++) {
                if (yaksuri_global.gpudriver[id].device[i].slab) {
                    yaksuri_global.gpudriver[id].info->gpu_free(yaksuri_global.gpudriver[id].
                                                                device[i].slab);
                }
            }
            free(yaksuri_global.gpudriver[id].device);

            rc = yaksuri_global.gpudriver[id].info->finalize();
            YAKSU_ERR_CHECK(rc, fn_fail);
            free(yaksuri_global.gpudriver[id].info);
        }
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_type_create_hook(yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_seq_type_create_hook(type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    for (yaksuri_gpudriver_id_e id = YAKSURI_GPUDRIVER_ID__UNSET;
         id < YAKSURI_GPUDRIVER_ID__LAST; id++) {
        if (id == YAKSURI_GPUDRIVER_ID__UNSET)
            continue;

        if (yaksuri_global.gpudriver[id].info) {
            rc = yaksuri_global.gpudriver[id].info->type_create(type);
            YAKSU_ERR_CHECK(rc, fn_fail);
        }
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_type_free_hook(yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_seq_type_free_hook(type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    for (yaksuri_gpudriver_id_e id = YAKSURI_GPUDRIVER_ID__UNSET;
         id < YAKSURI_GPUDRIVER_ID__LAST; id++) {
        if (id == YAKSURI_GPUDRIVER_ID__UNSET)
            continue;

        if (yaksuri_global.gpudriver[id].info) {
            rc = yaksuri_global.gpudriver[id].info->type_free(type);
            YAKSU_ERR_CHECK(rc, fn_fail);
        }
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_request_create_hook(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;

    request->backend.priv = malloc(sizeof(yaksuri_request_s));
    yaksuri_request_s *backend = (yaksuri_request_s *) request->backend.priv;

    backend->event = NULL;
    backend->kind = YAKSURI_REQUEST_KIND__UNSET;

    return rc;
}

int yaksur_request_free_hook(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_request_s *backend = (yaksuri_request_s *) request->backend.priv;
    yaksuri_gpudriver_id_e id = backend->gpudriver_id;

    if (backend->event) {
        assert(yaksuri_global.gpudriver[id].info);
        rc = yaksuri_global.gpudriver[id].info->event_destroy(backend->event);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    free(backend);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_info_create_hook(yaksi_info_s * info)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_seq_info_create_hook(info);
    YAKSU_ERR_CHECK(rc, fn_fail);

    for (yaksuri_gpudriver_id_e id = YAKSURI_GPUDRIVER_ID__UNSET;
         id < YAKSURI_GPUDRIVER_ID__LAST; id++) {
        if (id == YAKSURI_GPUDRIVER_ID__UNSET)
            continue;

        if (yaksuri_global.gpudriver[id].info) {
            rc = yaksuri_global.gpudriver[id].info->info_create(info);
            YAKSU_ERR_CHECK(rc, fn_fail);
        }
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_info_free_hook(yaksi_info_s * info)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_seq_info_free_hook(info);
    YAKSU_ERR_CHECK(rc, fn_fail);

    for (yaksuri_gpudriver_id_e id = YAKSURI_GPUDRIVER_ID__UNSET;
         id < YAKSURI_GPUDRIVER_ID__LAST; id++) {
        if (id == YAKSURI_GPUDRIVER_ID__UNSET)
            continue;

        if (yaksuri_global.gpudriver[id].info) {
            rc = yaksuri_global.gpudriver[id].info->info_free(info);
            YAKSU_ERR_CHECK(rc, fn_fail);
        }
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_info_keyval_append(yaksi_info_s * info, const char *key, const void *val,
                              unsigned int vallen)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_seq_info_keyval_append(info, key, val, vallen);
    YAKSU_ERR_CHECK(rc, fn_fail);

    for (yaksuri_gpudriver_id_e id = YAKSURI_GPUDRIVER_ID__UNSET;
         id < YAKSURI_GPUDRIVER_ID__LAST; id++) {
        if (id == YAKSURI_GPUDRIVER_ID__UNSET)
            continue;

        if (yaksuri_global.gpudriver[id].info) {
            rc = yaksuri_global.gpudriver[id].info->info_keyval_append(info, key, val, vallen);
            YAKSU_ERR_CHECK(rc, fn_fail);
        }
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
