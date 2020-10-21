/*
* Copyright (C) by Argonne National Laboratory
*     See COPYRIGHT in top-level directory
*/

#include <string.h>
#include "yaksi.h"
#include "yaksuri_seqi.h"
#include <assert.h>
#include <stdlib.h>

#define MAX_IOV_LENGTH (16384)

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

int yaksuri_seq_ipack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                      yaksi_info_s * info)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_seqi_type_s *seq_type = (yaksuri_seqi_type_s *) type->backend.seq.priv;

    uintptr_t iov_pack_threshold = YAKSURI_SEQI_INFO__DEFAULT_IOV_PUP_THRESHOLD;
    if (info) {
        yaksuri_seqi_info_s *seq_info = (yaksuri_seqi_info_s *) info->backend.seq.priv;
        iov_pack_threshold = seq_info->iov_pack_threshold;
    }

    if (type->is_contig) {
        memcpy(outbuf, (const char *) inbuf + type->true_lb, type->size * count);
    } else if (type->size / type->num_contig >= iov_pack_threshold) {
        struct iovec *iov;
        uintptr_t actual_iov_len;

        if (type->num_contig * count <= MAX_IOV_LENGTH) {
            iov = (struct iovec *) malloc(type->num_contig * count * sizeof(struct iovec));

            rc = yaksi_iov(inbuf, count, type, 0, iov, MAX_IOV_LENGTH, &actual_iov_len);
            YAKSU_ERR_CHECK(rc, fn_fail);
            assert(actual_iov_len == type->num_contig * count);

            char *dbuf = (char *) outbuf;
            for (uintptr_t i = 0; i < actual_iov_len; i++) {
                memcpy(dbuf, iov[i].iov_base, iov[i].iov_len);
                dbuf += iov[i].iov_len;
            }

            free(iov);
        } else if (type->num_contig <= MAX_IOV_LENGTH) {
            iov = (struct iovec *) malloc(type->num_contig * sizeof(struct iovec));

            uintptr_t iov_offset = 0;
            char *dbuf = (char *) outbuf;
            const char *sbuf = (const char *) inbuf;
            for (uintptr_t i = 0; i < count; i++) {
                rc = yaksi_iov(sbuf, 1, type, iov_offset, iov, MAX_IOV_LENGTH, &actual_iov_len);
                YAKSU_ERR_CHECK(rc, fn_fail);
                assert(actual_iov_len == type->num_contig);

                for (uintptr_t j = 0; j < actual_iov_len; j++) {
                    memcpy(dbuf, iov[j].iov_base, iov[j].iov_len);
                    dbuf += iov[j].iov_len;
                }

                sbuf += type->extent;
            }

            free(iov);
        } else {
            rc = YAKSA_ERR__NOT_SUPPORTED;
        }
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

int yaksuri_seq_iunpack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                        yaksi_info_s * info)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_seqi_type_s *seq_type = (yaksuri_seqi_type_s *) type->backend.seq.priv;

    uintptr_t iov_unpack_threshold = YAKSURI_SEQI_INFO__DEFAULT_IOV_PUP_THRESHOLD;
    if (info) {
        yaksuri_seqi_info_s *seq_info = (yaksuri_seqi_info_s *) info->backend.seq.priv;
        iov_unpack_threshold = seq_info->iov_unpack_threshold;
    }

    if (type->is_contig) {
        memcpy((char *) outbuf + type->true_lb, inbuf, type->size * count);
    } else if (type->size / type->num_contig >= iov_unpack_threshold) {
        struct iovec *iov;
        uintptr_t actual_iov_len;

        if (type->num_contig * count <= MAX_IOV_LENGTH) {
            iov = (struct iovec *) malloc(type->num_contig * count * sizeof(struct iovec));

            rc = yaksi_iov(outbuf, count, type, 0, iov, MAX_IOV_LENGTH, &actual_iov_len);
            YAKSU_ERR_CHECK(rc, fn_fail);
            assert(actual_iov_len == type->num_contig * count);

            const char *sbuf = (const char *) inbuf;
            for (uintptr_t i = 0; i < actual_iov_len; i++) {
                memcpy(iov[i].iov_base, sbuf, iov[i].iov_len);
                sbuf += iov[i].iov_len;
            }

            free(iov);
        } else if (type->num_contig <= MAX_IOV_LENGTH) {
            iov = (struct iovec *) malloc(type->num_contig * sizeof(struct iovec));

            uintptr_t iov_offset = 0;
            char *dbuf = (char *) outbuf;
            const char *sbuf = (const char *) inbuf;
            for (uintptr_t i = 0; i < count; i++) {
                rc = yaksi_iov(dbuf, 1, type, iov_offset, iov, MAX_IOV_LENGTH, &actual_iov_len);
                YAKSU_ERR_CHECK(rc, fn_fail);
                assert(actual_iov_len == type->num_contig);

                for (uintptr_t j = 0; j < actual_iov_len; j++) {
                    memcpy(iov[j].iov_base, sbuf, iov[j].iov_len);
                    sbuf += iov[j].iov_len;
                }

                dbuf += type->extent;
            }

            free(iov);
        } else {
            rc = YAKSA_ERR__NOT_SUPPORTED;
        }
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
