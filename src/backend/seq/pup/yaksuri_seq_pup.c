/*
* Copyright (C) by Argonne National Laboratory
*     See COPYRIGHT in top-level directory
*/

#include <string.h>
#include "yaksi.h"
#include "yaksuri_seqi.h"

int yaksuri_seq_pup_is_supported(yaksi_type_s * type, bool * is_supported)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_seqi_type_s *seq_type = (yaksuri_seqi_type_s *) type->backend.seq.priv;

    if (seq_type->pack || type->is_contig)
        *is_supported = true;
    else
        *is_supported = false;

    return rc;
}

int yaksuri_seq_ipack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_seqi_type_s *seq_type = (yaksuri_seqi_type_s *) type->backend.seq.priv;

    if (type->is_contig) {
        memcpy(outbuf, (const char *) inbuf + type->true_lb, type->size * count);
    } else if (seq_type->pack) {
        rc = seq_type->pack(inbuf, outbuf, count, type);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        rc = YAKSA_ERR__NOT_SUPPORTED;
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_seq_iunpack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_seqi_type_s *seq_type = (yaksuri_seqi_type_s *) type->backend.seq.priv;

    if (type->is_contig) {
        memcpy((char *) outbuf + type->true_lb, inbuf, type->size * count);
    } else if (seq_type->unpack) {
        rc = seq_type->unpack(inbuf, outbuf, count, type);
        YAKSU_ERR_CHECK(rc, fn_fail);
    } else {
        rc = YAKSA_ERR__NOT_SUPPORTED;
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
