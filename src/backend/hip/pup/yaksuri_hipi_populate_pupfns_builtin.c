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

int yaksuri_hipi_populate_pupfns_builtin(yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_hipi_type_s *hip = (yaksuri_hipi_type_s *) type->backend.hip.priv;

    switch (type->u.builtin.handle) {
        case YAKSA_TYPE__CHAR:
            hip->pack = yaksuri_hipi_pack_char;
            hip->unpack = yaksuri_hipi_unpack_char;
            break;
        case YAKSA_TYPE__WCHAR_T:
            hip->pack = yaksuri_hipi_pack_wchar_t;
            hip->unpack = yaksuri_hipi_unpack_wchar_t;
            break;
        case YAKSA_TYPE__INT8_T:
            hip->pack = yaksuri_hipi_pack_int8_t;
            hip->unpack = yaksuri_hipi_unpack_int8_t;
            break;
        case YAKSA_TYPE__INT16_T:
            hip->pack = yaksuri_hipi_pack_int16_t;
            hip->unpack = yaksuri_hipi_unpack_int16_t;
            break;
        case YAKSA_TYPE__INT32_T:
            hip->pack = yaksuri_hipi_pack_int32_t;
            hip->unpack = yaksuri_hipi_unpack_int32_t;
            break;
        case YAKSA_TYPE__INT64_T:
            hip->pack = yaksuri_hipi_pack_int64_t;
            hip->unpack = yaksuri_hipi_unpack_int64_t;
            break;
        case YAKSA_TYPE__FLOAT:
            hip->pack = yaksuri_hipi_pack_float;
            hip->unpack = yaksuri_hipi_unpack_float;
            break;
        case YAKSA_TYPE__DOUBLE:
            hip->pack = yaksuri_hipi_pack_double;
            hip->unpack = yaksuri_hipi_unpack_double;
            break;
        case YAKSA_TYPE__LONG_DOUBLE:
            hip->pack = yaksuri_hipi_pack_double;
            hip->unpack = yaksuri_hipi_unpack_double;
            break;
        default:
            break;
    }

    return rc;
}
