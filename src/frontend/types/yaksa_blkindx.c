/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include <stdlib.h>
#include <assert.h>

int yaksi_create_hindexed_block(int count, int blocklength, const intptr_t * array_of_displs,
                                yaksi_type_s * intype, yaksi_type_s ** newtype)
{
    int rc = YAKSA_SUCCESS;

    yaksi_type_s *outtype;
    rc = yaksi_type_alloc(&outtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    yaksu_atomic_incr(&intype->refcount);

    outtype->kind = YAKSI_TYPE_KIND__BLKHINDX;
    outtype->tree_depth = intype->tree_depth + 1;
    outtype->size = intype->size * blocklength * count;
    outtype->alignment = intype->alignment;

    intptr_t min_disp = array_of_displs[0];
    intptr_t max_disp = array_of_displs[0];
    for (int i = 1; i < count; i++) {
        if (array_of_displs[i] < min_disp)
            min_disp = array_of_displs[i];
        if (array_of_displs[i] > max_disp)
            max_disp = array_of_displs[i];
    }

    outtype->lb = min_disp + intype->lb;
    outtype->ub = max_disp + intype->lb + blocklength * intype->extent;
    outtype->true_lb = outtype->lb + intype->true_lb - intype->lb;
    outtype->true_ub = outtype->ub - intype->ub + intype->true_ub;
    outtype->extent = outtype->ub - outtype->lb;

    /* detect if the outtype is contiguous */
    if (intype->is_contig && outtype->ub == outtype->size) {
        outtype->is_contig = true;
        uintptr_t expected_disp = 0;
        for (int i = 0; i < count; i++) {
            if (array_of_displs[i] != expected_disp) {
                outtype->is_contig = false;
                break;
            }
            expected_disp = blocklength * intype->extent;
        }
    } else {
        outtype->is_contig = false;
    }

    if (outtype->is_contig) {
        outtype->num_contig = 1;
    } else if (intype->is_contig) {
        outtype->num_contig = intype->num_contig * count;
    } else {
        outtype->num_contig = intype->num_contig * count * blocklength;
    }

    outtype->u.blkhindx.count = count;
    outtype->u.blkhindx.blocklength = blocklength;
    outtype->u.blkhindx.array_of_displs = (intptr_t *) malloc(count * sizeof(intptr_t));
    for (int i = 0; i < count; i++)
        outtype->u.blkhindx.array_of_displs[i] = array_of_displs[i];
    outtype->u.blkhindx.child = intype;

    yaksur_type_create_hook(outtype);
    *newtype = outtype;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksa_create_hindexed_block(int count, int blocklength, const intptr_t * array_of_displs,
                                yaksa_type_t oldtype, yaksa_type_t * newtype)
{
    int rc = YAKSA_SUCCESS;

    assert(yaksi_global.is_initialized);

    yaksi_type_s *intype;
    rc = yaksi_type_get(oldtype, &intype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (count * intype->size == 0) {
        *newtype = YAKSA_TYPE__NULL;
        goto fn_exit;
    }

    yaksi_type_s *outtype;
    rc = yaksi_create_hindexed_block(count, blocklength, array_of_displs, intype, &outtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    *newtype = outtype->id;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksa_create_indexed_block(int count, int blocklength, const int *array_of_displs,
                               yaksa_type_t oldtype, yaksa_type_t * newtype)
{
    intptr_t *real_array_of_displs = NULL;
    int rc = YAKSA_SUCCESS;

    assert(yaksi_global.is_initialized);

    yaksi_type_s *intype;
    rc = yaksi_type_get(oldtype, &intype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (count * intype->size == 0) {
        *newtype = YAKSA_TYPE__NULL;
        goto fn_exit;
    }
    assert(count > 0);

    real_array_of_displs = (intptr_t *) malloc(count * sizeof(intptr_t));

    for (int i = 0; i < count; i++)
        real_array_of_displs[i] = array_of_displs[i] * intype->extent;

    yaksi_type_s *outtype;
    rc = yaksi_create_hindexed_block(count, blocklength, real_array_of_displs, intype, &outtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    *newtype = outtype->id;

  fn_exit:
    if (real_array_of_displs)
        free(real_array_of_displs);
    return rc;
  fn_fail:
    goto fn_exit;
}
