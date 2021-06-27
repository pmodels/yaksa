/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * DO NOT EDIT: AUTOMATICALLY GENERATED FILE !!
 */

#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri_hipi.h"
#include "yaksuri_hipi_populate_pupfns.h"
#include "yaksuri_hipi_pup.h"

int yaksuri_hipi_populate_pupfns_blkhindx_builtin(yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_hipi_type_s *hip = (yaksuri_hipi_type_s *) type->backend.hip.priv;

    char *str = getenv("YAKSA_ENV_MAX_NESTING_LEVEL");
    int max_nesting_level;
    if (str) {
        max_nesting_level = atoi(str);
    } else {
        max_nesting_level = YAKSI_ENV_DEFAULT_NESTING_LEVEL;
    }

    switch (type->u.blkhindx.child->u.builtin.handle) {
        case YAKSA_TYPE__CHAR:
            if (max_nesting_level >= 1) {
                hip->pack = yaksuri_hipi_pack_blkhindx_char;
                hip->unpack = yaksuri_hipi_unpack_blkhindx_char;
                hip->name = "yaksuri_hipi_op_blkhindx_char";
            }
            break;
        case YAKSA_TYPE__WCHAR_T:
            if (max_nesting_level >= 1) {
                hip->pack = yaksuri_hipi_pack_blkhindx_wchar_t;
                hip->unpack = yaksuri_hipi_unpack_blkhindx_wchar_t;
                hip->name = "yaksuri_hipi_op_blkhindx_wchar_t";
            }
            break;
        case YAKSA_TYPE__INT8_T:
            if (max_nesting_level >= 1) {
                hip->pack = yaksuri_hipi_pack_blkhindx_int8_t;
                hip->unpack = yaksuri_hipi_unpack_blkhindx_int8_t;
                hip->name = "yaksuri_hipi_op_blkhindx_int8_t";
            }
            break;
        case YAKSA_TYPE__INT16_T:
            if (max_nesting_level >= 1) {
                hip->pack = yaksuri_hipi_pack_blkhindx_int16_t;
                hip->unpack = yaksuri_hipi_unpack_blkhindx_int16_t;
                hip->name = "yaksuri_hipi_op_blkhindx_int16_t";
            }
            break;
        case YAKSA_TYPE__INT32_T:
            if (max_nesting_level >= 1) {
                hip->pack = yaksuri_hipi_pack_blkhindx_int32_t;
                hip->unpack = yaksuri_hipi_unpack_blkhindx_int32_t;
                hip->name = "yaksuri_hipi_op_blkhindx_int32_t";
            }
            break;
        case YAKSA_TYPE__INT64_T:
            if (max_nesting_level >= 1) {
                hip->pack = yaksuri_hipi_pack_blkhindx_int64_t;
                hip->unpack = yaksuri_hipi_unpack_blkhindx_int64_t;
                hip->name = "yaksuri_hipi_op_blkhindx_int64_t";
            }
            break;
        case YAKSA_TYPE__FLOAT:
            if (max_nesting_level >= 1) {
                hip->pack = yaksuri_hipi_pack_blkhindx_float;
                hip->unpack = yaksuri_hipi_unpack_blkhindx_float;
                hip->name = "yaksuri_hipi_op_blkhindx_float";
            }
            break;
        case YAKSA_TYPE__DOUBLE:
            if (max_nesting_level >= 1) {
                hip->pack = yaksuri_hipi_pack_blkhindx_double;
                hip->unpack = yaksuri_hipi_unpack_blkhindx_double;
                hip->name = "yaksuri_hipi_op_blkhindx_double";
            }
            break;
        case YAKSA_TYPE__LONG_DOUBLE:
            if (max_nesting_level >= 1) {
                hip->pack = yaksuri_hipi_pack_blkhindx_double;
                hip->unpack = yaksuri_hipi_unpack_blkhindx_double;
                hip->name = "yaksuri_hipi_op_blkhindx_double";
            }
            break;
        default:
            break;
    }

    return rc;
}
