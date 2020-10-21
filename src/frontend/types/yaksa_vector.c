/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include <stdlib.h>
#include <assert.h>

int yaksi_type_create_hvector(int count, int blocklength, intptr_t stride, yaksi_type_s * intype,
                              yaksi_type_s ** newtype)
{
    int rc = YAKSA_SUCCESS;

    /* shortcut for contig types */
    if (count == 1) {
        rc = yaksi_type_create_contig(blocklength, intype, newtype);
        YAKSU_ERR_CHECK(rc, fn_fail);
        goto fn_exit;
    }

    yaksi_type_s *outtype;
    outtype = (yaksi_type_s *) malloc(sizeof(yaksi_type_s));
    YAKSU_ERR_CHKANDJUMP(!outtype, rc, YAKSA_ERR__OUT_OF_MEM, fn_fail);
    yaksu_atomic_store(&outtype->refcount, 1);

    yaksu_atomic_incr(&intype->refcount);

    outtype->kind = YAKSI_TYPE_KIND__HVECTOR;
    outtype->tree_depth = intype->tree_depth + 1;
    outtype->size = intype->size * blocklength * count;
    outtype->alignment = intype->alignment;

    intptr_t min_disp;
    intptr_t max_disp;
    if (stride > 0) {
        min_disp = 0;
        max_disp = stride * (count - 1);
    } else {
        min_disp = stride * (count - 1);
        max_disp = 0;
    }

    outtype->lb = min_disp + intype->lb;
    outtype->ub = max_disp + intype->lb + blocklength * intype->extent;
    outtype->true_lb = outtype->lb + intype->true_lb - intype->lb;
    outtype->true_ub = outtype->ub - intype->ub + intype->true_ub;
    outtype->extent = outtype->ub - outtype->lb;

    /* detect if the outtype is contiguous */
    if (intype->is_contig && outtype->ub == outtype->size) {
        outtype->is_contig = true;
    } else {
        outtype->is_contig = false;
    }

    if (outtype->is_contig) {
        outtype->num_contig = 1;
    } else if (intype->is_contig) {
        outtype->num_contig = count * intype->num_contig;
    } else {
        outtype->num_contig = count * blocklength * intype->num_contig;
    }

    outtype->u.hvector.count = count;
    outtype->u.hvector.blocklength = blocklength;
    outtype->u.hvector.stride = stride;
    outtype->u.hvector.child = intype;

    yaksur_type_create_hook(outtype);
    *newtype = outtype;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksa_type_create_hvector(int count, int blocklength, intptr_t stride, yaksa_type_t oldtype,
                              yaksa_info_t info, yaksa_type_t * newtype)
{
    int rc = YAKSA_SUCCESS;

    assert(yaksu_atomic_load(&yaksi_is_initialized));

    yaksi_type_s *intype;
    rc = yaksi_type_get(oldtype, &intype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (count * intype->size == 0) {
        *newtype = YAKSA_TYPE__NULL;
        goto fn_exit;
    }

    yaksi_type_s *outtype;
    rc = yaksi_type_create_hvector(count, blocklength, stride, intype, &outtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksi_type_handle_alloc(outtype, newtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksa_type_create_vector(int count, int blocklength, int stride, yaksa_type_t oldtype,
                             yaksa_info_t info, yaksa_type_t * newtype)
{
    int rc = YAKSA_SUCCESS;

    assert(yaksu_atomic_load(&yaksi_is_initialized));

    yaksi_type_s *intype;
    rc = yaksi_type_get(oldtype, &intype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    if (count * intype->size == 0) {
        *newtype = YAKSA_TYPE__NULL;
        goto fn_exit;
    }

    yaksi_type_s *outtype;
    rc = yaksi_type_create_hvector(count, blocklength, (intptr_t) stride * intype->extent, intype,
                                   &outtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksi_type_handle_alloc(outtype, newtype);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
