/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <assert.h>
#include "yaksa.h"
#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri.h"

int yaksur_request_test(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_request_s *backend = (yaksuri_request_s *) request->backend.priv;
    yaksuri_gpudev_id_e id = backend->gpudev_id;

    if (backend->kind == YAKSURI_REQUEST_KIND__DEVICE_NATIVE) {
        int completed;
        rc = yaksuri_global.gpudev[id].info->event_query(backend->event, &completed);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if (completed) {
            yaksu_atomic_decr(&request->cc);
        }
    } else {
        rc = yaksuri_progress_poke();
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksur_request_wait(yaksi_request_s * request)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_request_s *backend = (yaksuri_request_s *) request->backend.priv;
    yaksuri_gpudev_id_e id = backend->gpudev_id;

    if (backend->kind == YAKSURI_REQUEST_KIND__DEVICE_NATIVE) {
        rc = yaksuri_global.gpudev[id].info->event_synchronize(backend->event);
        YAKSU_ERR_CHECK(rc, fn_fail);

        yaksu_atomic_decr(&request->cc);
        assert(!yaksu_atomic_load(&request->cc));
    } else {
        while (yaksu_atomic_load(&request->cc)) {
            rc = yaksuri_progress_poke();
            YAKSU_ERR_CHECK(rc, fn_fail);
        }
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
