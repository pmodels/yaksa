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
    yaksuri_gpudev_id_e id;

    rc = yaksuri_seq_init_hook();
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* CUDA hooks */
    id = YAKSURI_GPUDEV_ID__CUDA;
    yaksuri_global.gpudev[id].info = NULL;
    rc = yaksuri_cuda_init_hook(&yaksuri_global.gpudev[id].info);
    YAKSU_ERR_CHECK(rc, fn_fail);


    /* final setup for all backends */
    for (id = YAKSURI_GPUDEV_ID__UNSET + 1; id < YAKSURI_GPUDEV_ID__LAST; id++) {
        if (yaksuri_global.gpudev[id].info) {
            yaksuri_global.gpudev[id].host.slab = NULL;
            yaksuri_global.gpudev[id].host.slab_head_offset = 0;
            yaksuri_global.gpudev[id].host.slab_tail_offset = 0;

            int ndevices;
            rc = yaksuri_global.gpudev[id].info->get_num_devices(&ndevices);
            YAKSU_ERR_CHECK(rc, fn_fail);

            yaksuri_global.gpudev[id].device = (yaksuri_slab_s *)
                malloc(ndevices * sizeof(yaksuri_slab_s));
            for (int i = 0; i < ndevices; i++) {
                yaksuri_global.gpudev[id].device[i].slab = NULL;
                yaksuri_global.gpudev[id].device[i].slab_head_offset = 0;
                yaksuri_global.gpudev[id].device[i].slab_tail_offset = 0;
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

    for (yaksuri_gpudev_id_e id = YAKSURI_GPUDEV_ID__UNSET + 1; id < YAKSURI_GPUDEV_ID__LAST; id++) {
        if (yaksuri_global.gpudev[id].info) {
            if (yaksuri_global.gpudev[id].host.slab) {
                yaksuri_global.gpudev[id].info->host_free(yaksuri_global.gpudev[id].host.slab);
            }

            int ndevices;
            rc = yaksuri_global.gpudev[id].info->get_num_devices(&ndevices);
            YAKSU_ERR_CHECK(rc, fn_fail);

            for (int i = 0; i < ndevices; i++) {
                if (yaksuri_global.gpudev[id].device[i].slab) {
                    yaksuri_global.gpudev[id].info->device_free(yaksuri_global.gpudev[id].
                                                                device[i].slab);
                }
            }
            free(yaksuri_global.gpudev[id].device);

            rc = yaksuri_global.gpudev[id].info->finalize();
            YAKSU_ERR_CHECK(rc, fn_fail);
            free(yaksuri_global.gpudev[id].info);
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

    for (yaksuri_gpudev_id_e id = YAKSURI_GPUDEV_ID__UNSET + 1; id < YAKSURI_GPUDEV_ID__LAST; id++) {
        if (yaksuri_global.gpudev[id].info) {
            rc = yaksuri_global.gpudev[id].info->type_create(type);
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

    for (yaksuri_gpudev_id_e id = YAKSURI_GPUDEV_ID__UNSET + 1; id < YAKSURI_GPUDEV_ID__LAST; id++) {
        if (yaksuri_global.gpudev[id].info) {
            rc = yaksuri_global.gpudev[id].info->type_free(type);
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

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_request_free_hook(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_request_s *backend = (yaksuri_request_s *) request->backend.priv;
    yaksuri_gpudev_id_e id = backend->gpudev_id;

    if (backend->event) {
        assert(yaksuri_global.gpudev[id].info);
        rc = yaksuri_global.gpudev[id].info->event_destroy(backend->event);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    free(backend);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
