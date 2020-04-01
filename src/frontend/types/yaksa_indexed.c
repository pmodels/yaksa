/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include <stdlib.h>
#include <assert.h>

int yaksi_create_hindexed(int count, const int *array_of_blocklengths,
                          const intptr_t * array_of_displs, yaksi_type_s * intype,
                          yaksi_type_s ** newtype)
{
    int rc = YAKSA_SUCCESS;

    yaksi_type_s *outtype;
    rc = yaksi_type_alloc(&outtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    outtype->refcount = 1;
    yaksu_atomic_incr(&intype->refcount);

    outtype->kind = YAKSI_TYPE_KIND__HINDEXED;
    outtype->tree_depth = intype->tree_depth + 1;

    outtype->size = 0;
    for (int i = 0; i < count; i++)
        outtype->size += intype->size * array_of_blocklengths[i];

    int is_set = 0;
    for (int idx = 0; idx < count; idx++) {
        if (array_of_blocklengths[idx] == 0)
            continue;

        intptr_t lb = array_of_displs[idx] + intype->lb;
        intptr_t ub =
            array_of_displs[idx] + intype->lb + array_of_blocklengths[idx] * intype->extent;
        intptr_t true_lb = lb - intype->lb + intype->true_lb;
        intptr_t true_ub = ub - intype->ub + intype->true_ub;

        if (is_set) {
            outtype->lb = YAKSU_MIN(lb, outtype->lb);
            outtype->true_lb = YAKSU_MIN(true_lb, outtype->true_lb);
            outtype->ub = YAKSU_MAX(ub, outtype->ub);
            outtype->true_ub = YAKSU_MAX(true_ub, outtype->true_ub);
        } else {
            outtype->lb = lb;
            outtype->true_lb = true_lb;
            outtype->ub = ub;
            outtype->true_ub = true_ub;
        }

        is_set = 1;
    }

    outtype->extent = outtype->ub - outtype->lb;

    outtype->u.hindexed.count = count;
    outtype->u.hindexed.array_of_blocklengths = (int *) malloc(count * sizeof(intptr_t));
    outtype->u.hindexed.array_of_displs = (intptr_t *) malloc(count * sizeof(intptr_t));
    for (int i = 0; i < count; i++) {
        outtype->u.hindexed.array_of_blocklengths[i] = array_of_blocklengths[i];
        outtype->u.hindexed.array_of_displs[i] = array_of_displs[i];
    }
    outtype->u.hindexed.child = intype;

    /* detect if the outtype is contiguous */
    if (intype->is_contig && outtype->ub == outtype->size) {
        outtype->is_contig = true;
        uintptr_t expected_disp = 0;
        for (int i = 0; i < count; i++) {
            if (array_of_displs[i] != expected_disp) {
                outtype->is_contig = false;
                break;
            }
            expected_disp = array_of_blocklengths[i] * intype->extent;
        }
    } else {
        outtype->is_contig = false;
    }

    if (outtype->is_contig) {
        outtype->num_contig = 1;
    } else if (intype->is_contig) {
        uintptr_t tmp = 0;
        for (int i = 0; i < count; i++)
            if (array_of_blocklengths[i])
                tmp++;
        outtype->num_contig = intype->num_contig * tmp;
    } else {
        uintptr_t tmp = 0;
        for (int i = 0; i < count; i++)
            tmp += array_of_blocklengths[i];
        outtype->num_contig = intype->num_contig * tmp;
    }

    yaksur_type_create_hook(outtype);
    *newtype = outtype;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksa_create_hindexed(int count, const int *array_of_blocklengths,
                          const intptr_t * array_of_displs, yaksa_type_t oldtype,
                          yaksa_type_t * newtype)
{
    int rc = YAKSA_SUCCESS;

    assert(yaksi_global.is_initialized);

    yaksi_type_s *intype;
    rc = yaksi_type_get(oldtype, &intype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    yaksi_type_s *outtype;
    rc = yaksi_create_hindexed(count, array_of_blocklengths, array_of_displs, intype, &outtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    *newtype = outtype->id;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksa_create_indexed(int count, const int *array_of_blocklengths, const int *array_of_displs,
                         yaksa_type_t oldtype, yaksa_type_t * newtype)
{
    int rc = YAKSA_SUCCESS;
    intptr_t *real_array_of_displs = (intptr_t *) malloc(count * sizeof(intptr_t));

    assert(yaksi_global.is_initialized);

    yaksi_type_s *intype;
    rc = yaksi_type_get(oldtype, &intype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    for (int i = 0; i < count; i++)
        real_array_of_displs[i] = array_of_displs[i] * intype->extent;

    yaksi_type_s *outtype;
    rc = yaksi_create_hindexed(count, array_of_blocklengths, real_array_of_displs, intype,
                               &outtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    free(real_array_of_displs);

    *newtype = outtype->id;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
