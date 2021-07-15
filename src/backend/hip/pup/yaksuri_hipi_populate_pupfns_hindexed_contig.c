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

int yaksuri_hipi_populate_pupfns_hindexed_contig(yaksi_type_s * type)
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

    switch (type->u.hindexed.child->u.contig.child->kind) {
        case YAKSI_TYPE_KIND__HVECTOR:
            switch (type->u.hindexed.child->u.contig.child->u.hvector.child->kind) {
                case YAKSI_TYPE_KIND__BUILTIN:
                    switch (type->u.hindexed.child->u.contig.child->u.hvector.child->u.
                            builtin.handle) {
                        case YAKSA_TYPE__CHAR:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hvector_char;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hvector_char;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hvector_char";
                            }
                            break;
                        case YAKSA_TYPE__WCHAR_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hvector_wchar_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hvector_wchar_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hvector_wchar_t";
                            }
                            break;
                        case YAKSA_TYPE__INT8_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hvector_int8_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hvector_int8_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hvector_int8_t";
                            }
                            break;
                        case YAKSA_TYPE__INT16_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hvector_int16_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hvector_int16_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hvector_int16_t";
                            }
                            break;
                        case YAKSA_TYPE__INT32_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hvector_int32_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hvector_int32_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hvector_int32_t";
                            }
                            break;
                        case YAKSA_TYPE__INT64_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hvector_int64_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hvector_int64_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hvector_int64_t";
                            }
                            break;
                        case YAKSA_TYPE__FLOAT:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hvector_float;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hvector_float;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hvector_float";
                            }
                            break;
                        case YAKSA_TYPE__DOUBLE:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hvector_double;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hvector_double;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hvector_double";
                            }
                            break;
                        case YAKSA_TYPE__LONG_DOUBLE:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hvector_double;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hvector_double;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hvector_double";
                            }
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
            break;
        case YAKSI_TYPE_KIND__BLKHINDX:
            switch (type->u.hindexed.child->u.contig.child->u.blkhindx.child->kind) {
                case YAKSI_TYPE_KIND__BUILTIN:
                    switch (type->u.hindexed.child->u.contig.child->u.blkhindx.child->u.
                            builtin.handle) {
                        case YAKSA_TYPE__CHAR:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_blkhindx_char;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_blkhindx_char;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_blkhindx_char";
                            }
                            break;
                        case YAKSA_TYPE__WCHAR_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_blkhindx_wchar_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_blkhindx_wchar_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_blkhindx_wchar_t";
                            }
                            break;
                        case YAKSA_TYPE__INT8_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_blkhindx_int8_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_blkhindx_int8_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_blkhindx_int8_t";
                            }
                            break;
                        case YAKSA_TYPE__INT16_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_blkhindx_int16_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_blkhindx_int16_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_blkhindx_int16_t";
                            }
                            break;
                        case YAKSA_TYPE__INT32_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_blkhindx_int32_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_blkhindx_int32_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_blkhindx_int32_t";
                            }
                            break;
                        case YAKSA_TYPE__INT64_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_blkhindx_int64_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_blkhindx_int64_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_blkhindx_int64_t";
                            }
                            break;
                        case YAKSA_TYPE__FLOAT:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_blkhindx_float;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_blkhindx_float;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_blkhindx_float";
                            }
                            break;
                        case YAKSA_TYPE__DOUBLE:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_blkhindx_double;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_blkhindx_double;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_blkhindx_double";
                            }
                            break;
                        case YAKSA_TYPE__LONG_DOUBLE:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_blkhindx_double;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_blkhindx_double;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_blkhindx_double";
                            }
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
            break;
        case YAKSI_TYPE_KIND__HINDEXED:
            switch (type->u.hindexed.child->u.contig.child->u.hindexed.child->kind) {
                case YAKSI_TYPE_KIND__BUILTIN:
                    switch (type->u.hindexed.child->u.contig.child->u.hindexed.child->u.
                            builtin.handle) {
                        case YAKSA_TYPE__CHAR:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hindexed_char;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hindexed_char;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hindexed_char";
                            }
                            break;
                        case YAKSA_TYPE__WCHAR_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hindexed_wchar_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hindexed_wchar_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hindexed_wchar_t";
                            }
                            break;
                        case YAKSA_TYPE__INT8_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hindexed_int8_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hindexed_int8_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hindexed_int8_t";
                            }
                            break;
                        case YAKSA_TYPE__INT16_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hindexed_int16_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hindexed_int16_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hindexed_int16_t";
                            }
                            break;
                        case YAKSA_TYPE__INT32_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hindexed_int32_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hindexed_int32_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hindexed_int32_t";
                            }
                            break;
                        case YAKSA_TYPE__INT64_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hindexed_int64_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hindexed_int64_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hindexed_int64_t";
                            }
                            break;
                        case YAKSA_TYPE__FLOAT:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hindexed_float;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hindexed_float;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hindexed_float";
                            }
                            break;
                        case YAKSA_TYPE__DOUBLE:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hindexed_double;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hindexed_double;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hindexed_double";
                            }
                            break;
                        case YAKSA_TYPE__LONG_DOUBLE:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_hindexed_double;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_hindexed_double;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_hindexed_double";
                            }
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
            break;
        case YAKSI_TYPE_KIND__CONTIG:
            switch (type->u.hindexed.child->u.contig.child->u.contig.child->kind) {
                case YAKSI_TYPE_KIND__BUILTIN:
                    switch (type->u.hindexed.child->u.contig.child->u.contig.child->u.
                            builtin.handle) {
                        case YAKSA_TYPE__CHAR:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_contig_char;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_contig_char;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_contig_char";
                            }
                            break;
                        case YAKSA_TYPE__WCHAR_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_contig_wchar_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_contig_wchar_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_contig_wchar_t";
                            }
                            break;
                        case YAKSA_TYPE__INT8_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_contig_int8_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_contig_int8_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_contig_int8_t";
                            }
                            break;
                        case YAKSA_TYPE__INT16_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_contig_int16_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_contig_int16_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_contig_int16_t";
                            }
                            break;
                        case YAKSA_TYPE__INT32_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_contig_int32_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_contig_int32_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_contig_int32_t";
                            }
                            break;
                        case YAKSA_TYPE__INT64_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_contig_int64_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_contig_int64_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_contig_int64_t";
                            }
                            break;
                        case YAKSA_TYPE__FLOAT:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_contig_float;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_contig_float;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_contig_float";
                            }
                            break;
                        case YAKSA_TYPE__DOUBLE:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_contig_double;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_contig_double;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_contig_double";
                            }
                            break;
                        case YAKSA_TYPE__LONG_DOUBLE:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_contig_double;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_contig_double;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_contig_double";
                            }
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
            break;
        case YAKSI_TYPE_KIND__RESIZED:
            switch (type->u.hindexed.child->u.contig.child->u.resized.child->kind) {
                case YAKSI_TYPE_KIND__BUILTIN:
                    switch (type->u.hindexed.child->u.contig.child->u.resized.child->u.
                            builtin.handle) {
                        case YAKSA_TYPE__CHAR:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_resized_char;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_resized_char;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_resized_char";
                            }
                            break;
                        case YAKSA_TYPE__WCHAR_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_resized_wchar_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_resized_wchar_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_resized_wchar_t";
                            }
                            break;
                        case YAKSA_TYPE__INT8_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_resized_int8_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_resized_int8_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_resized_int8_t";
                            }
                            break;
                        case YAKSA_TYPE__INT16_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_resized_int16_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_resized_int16_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_resized_int16_t";
                            }
                            break;
                        case YAKSA_TYPE__INT32_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_resized_int32_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_resized_int32_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_resized_int32_t";
                            }
                            break;
                        case YAKSA_TYPE__INT64_T:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_resized_int64_t;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_resized_int64_t;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_resized_int64_t";
                            }
                            break;
                        case YAKSA_TYPE__FLOAT:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_resized_float;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_resized_float;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_resized_float";
                            }
                            break;
                        case YAKSA_TYPE__DOUBLE:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_resized_double;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_resized_double;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_resized_double";
                            }
                            break;
                        case YAKSA_TYPE__LONG_DOUBLE:
                            if (max_nesting_level >= 3) {
                                hip->pack = yaksuri_hipi_pack_hindexed_contig_resized_double;
                                hip->unpack = yaksuri_hipi_unpack_hindexed_contig_resized_double;
                                hip->name = "yaksuri_hipi_op_hindexed_contig_resized_double";
                            }
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
            break;
        case YAKSI_TYPE_KIND__BUILTIN:
            switch (type->u.hindexed.child->u.contig.child->u.builtin.handle) {
                case YAKSA_TYPE__CHAR:
                    if (max_nesting_level >= 2) {
                        hip->pack = yaksuri_hipi_pack_hindexed_contig_char;
                        hip->unpack = yaksuri_hipi_unpack_hindexed_contig_char;
                        hip->name = "yaksuri_hipi_op_hindexed_contig_char";
                    }
                    break;
                case YAKSA_TYPE__WCHAR_T:
                    if (max_nesting_level >= 2) {
                        hip->pack = yaksuri_hipi_pack_hindexed_contig_wchar_t;
                        hip->unpack = yaksuri_hipi_unpack_hindexed_contig_wchar_t;
                        hip->name = "yaksuri_hipi_op_hindexed_contig_wchar_t";
                    }
                    break;
                case YAKSA_TYPE__INT8_T:
                    if (max_nesting_level >= 2) {
                        hip->pack = yaksuri_hipi_pack_hindexed_contig_int8_t;
                        hip->unpack = yaksuri_hipi_unpack_hindexed_contig_int8_t;
                        hip->name = "yaksuri_hipi_op_hindexed_contig_int8_t";
                    }
                    break;
                case YAKSA_TYPE__INT16_T:
                    if (max_nesting_level >= 2) {
                        hip->pack = yaksuri_hipi_pack_hindexed_contig_int16_t;
                        hip->unpack = yaksuri_hipi_unpack_hindexed_contig_int16_t;
                        hip->name = "yaksuri_hipi_op_hindexed_contig_int16_t";
                    }
                    break;
                case YAKSA_TYPE__INT32_T:
                    if (max_nesting_level >= 2) {
                        hip->pack = yaksuri_hipi_pack_hindexed_contig_int32_t;
                        hip->unpack = yaksuri_hipi_unpack_hindexed_contig_int32_t;
                        hip->name = "yaksuri_hipi_op_hindexed_contig_int32_t";
                    }
                    break;
                case YAKSA_TYPE__INT64_T:
                    if (max_nesting_level >= 2) {
                        hip->pack = yaksuri_hipi_pack_hindexed_contig_int64_t;
                        hip->unpack = yaksuri_hipi_unpack_hindexed_contig_int64_t;
                        hip->name = "yaksuri_hipi_op_hindexed_contig_int64_t";
                    }
                    break;
                case YAKSA_TYPE__FLOAT:
                    if (max_nesting_level >= 2) {
                        hip->pack = yaksuri_hipi_pack_hindexed_contig_float;
                        hip->unpack = yaksuri_hipi_unpack_hindexed_contig_float;
                        hip->name = "yaksuri_hipi_op_hindexed_contig_float";
                    }
                    break;
                case YAKSA_TYPE__DOUBLE:
                    if (max_nesting_level >= 2) {
                        hip->pack = yaksuri_hipi_pack_hindexed_contig_double;
                        hip->unpack = yaksuri_hipi_unpack_hindexed_contig_double;
                        hip->name = "yaksuri_hipi_op_hindexed_contig_double";
                    }
                    break;
                case YAKSA_TYPE__LONG_DOUBLE:
                    if (max_nesting_level >= 2) {
                        hip->pack = yaksuri_hipi_pack_hindexed_contig_double;
                        hip->unpack = yaksuri_hipi_unpack_hindexed_contig_double;
                        hip->name = "yaksuri_hipi_op_hindexed_contig_double";
                    }
                    break;
                default:
                    break;
            }
            break;
        default:
            break;
    }

    return rc;
}
