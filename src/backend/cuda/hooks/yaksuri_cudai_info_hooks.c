/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksuri_cudai.h"
#include <assert.h>
#include <string.h>
#include <stdlib.h>

int yaksuri_cudai_info_create_hook(yaksi_info_s * info)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_info_s *cuda;

    cuda = (yaksuri_cudai_info_s *) malloc(sizeof(yaksuri_cudai_info_s));

    /* set default values for info keys */
    cuda->iov_pack_threshold = YAKSURI_CUDAI_INFO__DEFAULT_IOV_PUP_THRESHOLD;
    cuda->iov_unpack_threshold = YAKSURI_CUDAI_INFO__DEFAULT_IOV_PUP_THRESHOLD;

    info->backend.cuda.priv = (void *) cuda;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cudai_info_free_hook(yaksi_info_s * info)
{
    int rc = YAKSA_SUCCESS;

    free(info->backend.cuda.priv);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cudai_info_keyval_append(yaksi_info_s * info, const char *key, const void *val,
                                     unsigned int vallen)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_info_s *cuda = (yaksuri_cudai_info_s *) info->backend.cuda.priv;

    if (!strncmp(key, "yaksa_cuda_iov_pack_threshold", YAKSA_INFO_MAX_KEYLEN)) {
        assert(vallen == sizeof(uintptr_t));
        cuda->iov_pack_threshold = (uintptr_t) val;
    } else if (!strncmp(key, "yaksa_cuda_iov_unpack_threshold", YAKSA_INFO_MAX_KEYLEN)) {
        assert(vallen == sizeof(uintptr_t));
        cuda->iov_unpack_threshold = (uintptr_t) val;
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
