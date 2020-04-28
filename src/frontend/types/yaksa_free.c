/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksa.h"
#include "yaksu.h"
#include "yaksi.h"
#include <stdlib.h>
#include <assert.h>

int yaksi_free(yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;

    int ret = yaksu_atomic_decr(&type->refcount);
    assert(ret >= 1);

    if (ret > 1)
        goto fn_exit;

    rc = yaksur_type_free_hook(type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* free the child types */
    switch (type->kind) {
        case YAKSI_TYPE_KIND__CONTIG:
            rc = yaksi_free(type->u.contig.child);
            YAKSU_ERR_CHECK(rc, fn_fail);
            break;

        case YAKSI_TYPE_KIND__DUP:
            rc = yaksi_free(type->u.dup.child);
            YAKSU_ERR_CHECK(rc, fn_fail);
            break;

        case YAKSI_TYPE_KIND__RESIZED:
            rc = yaksi_free(type->u.resized.child);
            YAKSU_ERR_CHECK(rc, fn_fail);
            break;

        case YAKSI_TYPE_KIND__HVECTOR:
            rc = yaksi_free(type->u.hvector.child);
            YAKSU_ERR_CHECK(rc, fn_fail);
            break;

        case YAKSI_TYPE_KIND__BLKHINDX:
            rc = yaksi_free(type->u.blkhindx.child);
            YAKSU_ERR_CHECK(rc, fn_fail);
            free(type->u.blkhindx.array_of_displs);
            break;

        case YAKSI_TYPE_KIND__HINDEXED:
            rc = yaksi_free(type->u.hindexed.child);
            YAKSU_ERR_CHECK(rc, fn_fail);
            free(type->u.hindexed.array_of_blocklengths);
            free(type->u.hindexed.array_of_displs);
            break;

        case YAKSI_TYPE_KIND__STRUCT:
            for (int i = 0; i < type->u.str.count; i++) {
                rc = yaksi_free(type->u.str.array_of_types[i]);
                YAKSU_ERR_CHECK(rc, fn_fail);
            }
            free(type->u.str.array_of_types);
            free(type->u.str.array_of_blocklengths);
            free(type->u.str.array_of_displs);
            break;

        case YAKSI_TYPE_KIND__SUBARRAY:
            rc = yaksi_free(type->u.subarray.primary);
            YAKSU_ERR_CHECK(rc, fn_fail);
            break;

        default:
            break;
    }

    yaksi_type_free(type);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksa_free(yaksa_type_t type)
{
    yaksi_type_s *yaksi_type;
    int rc = YAKSA_SUCCESS;

    assert(yaksi_global.is_initialized);

    if (type == YAKSA_TYPE__NULL)
        goto fn_exit;

    rc = yaksi_type_get(type, &yaksi_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksi_free(yaksi_type);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
