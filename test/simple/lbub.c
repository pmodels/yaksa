/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksa.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif

/*
   The default behavior of the test routines should be to briefly indicate
   the cause of any errors - in this test, that means that verbose needs
   to be set. Verbose should turn on output that is independent of error
   levels.
*/
static int verbose = 1;

/* tests */
int int_with_lb_ub_test(void);
int contig_of_int_with_lb_ub_test(void);
int contig_negextent_of_int_with_lb_ub_test(void);
int vector_of_int_with_lb_ub_test(void);
int vector_blklen_of_int_with_lb_ub_test(void);
int vector_blklen_stride_of_int_with_lb_ub_test(void);
int vector_blklen_stride_negextent_of_int_with_lb_ub_test(void);
int vector_blklen_negstride_negextent_of_int_with_lb_ub_test(void);
int int_with_negextent_test(void);
int vector_blklen_negstride_of_int_with_lb_ub_test(void);

static void TestPrintError(int err)
{
    fprintf(stderr, "Found %d errors\n", err);
}

int main(int argc, char **argv)
{
    int err, errs = 0;

    yaksa_init(NULL);

    /* perform some tests */
    err = int_with_lb_ub_test();
    if (err && verbose)
        fprintf(stderr, "found %d errors in simple lb/ub test\n", err);
    errs += err;

    err = contig_of_int_with_lb_ub_test();
    if (err && verbose)
        fprintf(stderr, "found %d errors in contig test\n", err);
    errs += err;

    err = contig_negextent_of_int_with_lb_ub_test();
    if (err && verbose)
        fprintf(stderr, "found %d errors in negextent contig test\n", err);
    errs += err;

    err = vector_of_int_with_lb_ub_test();
    if (err && verbose)
        fprintf(stderr, "found %d errors in simple vector test\n", err);
    errs += err;

    err = vector_blklen_of_int_with_lb_ub_test();
    if (err && verbose)
        fprintf(stderr, "found %d errors in vector blklen test\n", err);
    errs += err;

    err = vector_blklen_stride_of_int_with_lb_ub_test();
    if (err && verbose)
        fprintf(stderr, "found %d errors in strided vector test\n", err);
    errs += err;

    err = vector_blklen_negstride_of_int_with_lb_ub_test();
    if (err && verbose)
        fprintf(stderr, "found %d errors in negstrided vector test\n", err);
    errs += err;

    err = int_with_negextent_test();
    if (err && verbose)
        fprintf(stderr, "found %d errors in negextent lb/ub test\n", err);
    errs += err;

    err = vector_blklen_stride_negextent_of_int_with_lb_ub_test();
    if (err && verbose)
        fprintf(stderr, "found %d errors in strided negextent vector test\n", err);
    errs += err;

    err = vector_blklen_negstride_negextent_of_int_with_lb_ub_test();
    if (err && verbose)
        fprintf(stderr, "found %d errors in negstrided negextent vector test\n", err);
    errs += err;

    yaksa_finalize();
    return 0;
}

int int_with_lb_ub_test(void)
{
    int err, errs = 0, val;
    intptr_t lb, extent, aval, true_lb;

    yaksa_type_t tmptype, eviltype;

    err = yaksa_type_create_contig(4, YAKSA_TYPE__BYTE, NULL, &tmptype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_contig failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    err = yaksa_type_create_resized(tmptype, -3, 9, NULL, &eviltype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_resized failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    err = yaksa_type_get_size(eviltype, (uintptr_t *) (void *) &val);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_size failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (val != 4) {
        errs++;
        if (verbose)
            fprintf(stderr, "  size of type = %d; should be %d\n", val, 4);
    }

    err = yaksa_type_get_extent(eviltype, &lb, &extent);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (lb != -3) {
        errs++;
        if (verbose)
            fprintf(stderr, "  lb of type = %d; should be %d\n", (int) aval, -3);
    }

    if (extent != 9) {
        errs++;
        if (verbose)
            fprintf(stderr, "  extent of type = %d; should be %d\n", (int) extent, 9);
    }

    err = yaksa_type_get_true_extent(eviltype, &true_lb, &aval);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_true_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (true_lb != 0) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true_lb of type = %d; should be %d\n", (int) true_lb, 0);
    }

    if (aval != 4) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true extent of type = %d; should be %d\n", (int) aval, 4);
    }

    yaksa_type_free(tmptype);
    yaksa_type_free(eviltype);

    return errs;
}

int contig_of_int_with_lb_ub_test(void)
{
    int err, errs = 0, val;
    intptr_t lb, extent, aval, true_lb;
    char *typemapstring = 0;

    yaksa_type_t tmptype, inttype, eviltype;

    /* build same type as in int_with_lb_ub_test() */
    typemapstring = (char *) "{ 4*(BYTE,0) }";
    err = yaksa_type_create_contig(4, YAKSA_TYPE__BYTE, NULL, &tmptype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_contig of %s failed.\n", typemapstring);
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    typemapstring = (char *) "{ (LB,-3),4*(BYTE,0),(UB,6) }";
    err = yaksa_type_create_resized(tmptype, -3, 9, NULL, &inttype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_resized of %s failed.\n", typemapstring);
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    typemapstring = (char *)
        "{ (LB,-3),4*(BYTE,0),(UB,6),(LB,6),4*(BYTE,9),(UB,15),(LB,15),4*(BYTE,18),(UB,24)}";
    err = yaksa_type_create_contig(3, inttype, NULL, &eviltype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_contig of %s failed.\n", typemapstring);
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_get_size(eviltype, (uintptr_t *) (void *) &val);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_size of %s failed.\n", typemapstring);
        if (verbose)
            TestPrintError(err);
    }

    if (val != 12) {
        errs++;
        if (verbose)
            fprintf(stderr, "  size of type = %d; should be %d\n", val, 12);
    }

    err = yaksa_type_get_extent(eviltype, &lb, &extent);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (lb != -3) {
        errs++;
        if (verbose)
            fprintf(stderr, "  lb of type = %d from Type_get_extent; should be %d in %s\n",
                    (int) aval, -3, typemapstring);
    }

    if (extent != 27) {
        errs++;
        if (verbose)
            fprintf(stderr, "  extent of type = %d from Type_get_extent; should be %d in %s\n",
                    (int) extent, 27, typemapstring);
    }

    err = yaksa_type_get_true_extent(eviltype, &true_lb, &aval);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_true_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (true_lb != 0) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true_lb of type = %d; should be %d in %s\n", (int) true_lb, 0,
                    typemapstring);
    }

    if (aval != 22) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true extent of type = %d; should be %d in %s\n", (int) aval, 22,
                    typemapstring);
    }

    yaksa_type_free(tmptype);
    yaksa_type_free(inttype);
    yaksa_type_free(eviltype);

    return errs;
}

int contig_negextent_of_int_with_lb_ub_test(void)
{
    int err, errs = 0, val;
    intptr_t lb, extent, aval, true_lb;
    char *typemapstring = 0;

    yaksa_type_t tmptype, inttype, eviltype;

    /* build same type as in int_with_lb_ub_test() */
    typemapstring = (char *) "{ 4*(BYTE,0) }";
    err = yaksa_type_create_contig(4, YAKSA_TYPE__BYTE, NULL, &tmptype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_contig of %s failed.\n", typemapstring);
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    typemapstring = (char *) "{ (LB,6),4*(BYTE,0),(UB,-3) }";
    err = yaksa_type_create_resized(tmptype, 6, -9, NULL, &inttype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_resized of %s failed.\n", typemapstring);
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    typemapstring = (char *)
        "{ (LB,6),4*(BYTE,0),(UB,-3),(LB,-3),4*(BYTE,-9),(UB,-12),(LB,-12),4*(BYTE,-18),(UB,-21) }";
    err = yaksa_type_create_contig(3, inttype, NULL, &eviltype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_contig of %s failed.\n", typemapstring);
        if (verbose)
            TestPrintError(err);
        /* No point in continuing */
        return errs;
    }

    err = yaksa_type_get_size(eviltype, (uintptr_t *) (void *) &val);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_size of %s failed.\n", typemapstring);
        if (verbose)
            TestPrintError(err);
    }

    if (val != 12) {
        errs++;
        if (verbose)
            fprintf(stderr, "  size of type = %d; should be %d\n", val, 12);
    }

    err = yaksa_type_get_extent(eviltype, &lb, &extent);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (lb != -12) {
        errs++;
        if (verbose)
            fprintf(stderr, "  lb of type = %d; should be %d\n", (int) aval, -12);
    }

    if (extent != 9) {
        errs++;
        if (verbose)
            fprintf(stderr, "  extent of type = %d; should be %d\n", (int) extent, 9);
    }

    err = yaksa_type_get_true_extent(eviltype, &true_lb, &aval);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_true_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (true_lb != -18) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true_lb of type = %d; should be %d\n", (int) true_lb, -18);
    }

    if (aval != 22) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true extent of type = %d; should be %d\n", (int) aval, 22);
    }

    yaksa_type_free(tmptype);
    yaksa_type_free(inttype);
    yaksa_type_free(eviltype);

    return errs;
}

int vector_of_int_with_lb_ub_test(void)
{
    int err, errs = 0, val;
    intptr_t lb, extent, aval, true_lb;

    yaksa_type_t tmptype, inttype, eviltype;

    /* build same type as in int_with_lb_ub_test() */
    err = yaksa_type_create_contig(4, YAKSA_TYPE__BYTE, NULL, &tmptype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_contig failed.\n");
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_create_resized(tmptype, -3, 9, NULL, &inttype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_resized failed.\n");
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_create_vector(3, 1, 1, inttype, NULL, &eviltype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_vector failed.\n");
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_get_size(eviltype, (uintptr_t *) (void *) &val);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_size failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (val != 12) {
        errs++;
        if (verbose)
            fprintf(stderr, "  size of type = %d; should be %d\n", val, 12);
    }

    err = yaksa_type_get_extent(eviltype, &lb, &extent);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (lb != -3) {
        errs++;
        if (verbose)
            fprintf(stderr, "  lb of type = %d; should be %d\n", (int) aval, -3);
    }

    if (extent != 27) {
        errs++;
        if (verbose)
            fprintf(stderr, "  extent of type = %d; should be %d\n", (int) extent, 27);
    }

    err = yaksa_type_get_true_extent(eviltype, &true_lb, &aval);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_true_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (true_lb != 0) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true_lb of type = %d; should be %d\n", (int) true_lb, 0);
    }

    if (aval != 22) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true extent of type = %d; should be %d\n", (int) aval, 22);
    }

    yaksa_type_free(tmptype);
    yaksa_type_free(inttype);
    yaksa_type_free(eviltype);

    return errs;
}

/*
 * blklen = 4
 */
int vector_blklen_of_int_with_lb_ub_test(void)
{
    int err, errs = 0, val;
    intptr_t lb, extent, aval, true_lb;

    yaksa_type_t tmptype, inttype, eviltype;

    /* build same type as in int_with_lb_ub_test() */
    err = yaksa_type_create_contig(4, YAKSA_TYPE__BYTE, NULL, &tmptype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_contig failed.\n");
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_create_resized(tmptype, -3, 9, NULL, &inttype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_resized failed.\n");
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_create_vector(3, 4, 1, inttype, NULL, &eviltype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_vector failed.\n");
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_get_size(eviltype, (uintptr_t *) (void *) &val);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_size failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (val != 48) {
        errs++;
        if (verbose)
            fprintf(stderr, "  size of type = %d; should be %d\n", val, 48);
    }

    err = yaksa_type_get_extent(eviltype, &lb, &extent);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (lb != -3) {
        errs++;
        if (verbose)
            fprintf(stderr, "  lb of type = %d; should be %d\n", (int) aval, -3);
    }

    if (extent != 54) {
        errs++;
        if (verbose)
            fprintf(stderr, "  extent of type = %d; should be %d\n", (int) extent, 54);
    }

    err = yaksa_type_get_true_extent(eviltype, &true_lb, &aval);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_true_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (true_lb != 0) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true_lb of type = %d; should be %d\n", (int) true_lb, 0);
    }

    if (aval != 49) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true extent of type = %d; should be %d\n", (int) aval, 49);
    }

    yaksa_type_free(tmptype);
    yaksa_type_free(inttype);
    yaksa_type_free(eviltype);

    return errs;
}

int vector_blklen_stride_of_int_with_lb_ub_test(void)
{
    int err, errs = 0, val;
    intptr_t lb, extent, aval, true_lb;
    char *typemapstring = 0;

    yaksa_type_t tmptype, inttype, eviltype;

    /* build same type as in int_with_lb_ub_test() */
    typemapstring = (char *) "{ 4*(BYTE,0) }";
    err = yaksa_type_create_contig(4, YAKSA_TYPE__BYTE, NULL, &tmptype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_contig of %s failed.\n", typemapstring);
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    typemapstring = (char *) "{ (LB,-3),4*(BYTE,0),(UB,6) }";
    err = yaksa_type_create_resized(tmptype, -3, 9, NULL, &inttype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_resized of %s failed.\n", typemapstring);
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_create_vector(3, 4, 5, inttype, NULL, &eviltype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_vector failed.\n");
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_get_size(eviltype, (uintptr_t *) (void *) &val);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_size failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (val != 48) {
        errs++;
        if (verbose)
            fprintf(stderr, "  size of type = %d; should be %d\n", val, 48);
    }

    err = yaksa_type_get_extent(eviltype, &lb, &extent);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (lb != -3) {
        errs++;
        if (verbose)
            fprintf(stderr, "  lb of type = %d; should be %d\n", (int) aval, -3);
    }

    if (extent != 126) {
        errs++;
        if (verbose)
            fprintf(stderr, "  extent of type = %d; should be %d\n", (int) extent, 126);
    }

    err = yaksa_type_get_true_extent(eviltype, &true_lb, &aval);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_true_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (true_lb != 0) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true_lb of type = %d; should be %d\n", (int) true_lb, 0);
    }

    if (aval != 121) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true extent of type = %d; should be %d\n", (int) aval, 121);
    }

    yaksa_type_free(tmptype);
    yaksa_type_free(inttype);
    yaksa_type_free(eviltype);

    return errs;
}

int vector_blklen_negstride_of_int_with_lb_ub_test(void)
{
    int err, errs = 0, val;
    intptr_t lb, extent, aval, true_lb;

    yaksa_type_t tmptype, inttype, eviltype;

    /* build same type as in int_with_lb_ub_test() */
    err = yaksa_type_create_contig(4, YAKSA_TYPE__BYTE, NULL, &tmptype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_contig failed.\n");
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_create_resized(tmptype, -3, 9, NULL, &inttype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_resized failed.\n");
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_create_vector(3, 4, -5, inttype, NULL, &eviltype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_vector failed.\n");
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_get_size(eviltype, (uintptr_t *) (void *) &val);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_size failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (val != 48) {
        errs++;
        if (verbose)
            fprintf(stderr, "  size of type = %d; should be %d\n", val, 48);
    }

    err = yaksa_type_get_extent(eviltype, &lb, &extent);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (lb != -93) {
        errs++;
        if (verbose)
            fprintf(stderr, "  lb of type = %d; should be %d\n", (int) aval, -93);
    }

    if (extent != 126) {
        errs++;
        if (verbose)
            fprintf(stderr, "  extent of type = %d; should be %d\n", (int) extent, 126);
    }

    err = yaksa_type_get_true_extent(eviltype, &true_lb, &aval);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_true_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (true_lb != -90) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true_lb of type = %d; should be %d\n", (int) true_lb, -90);
    }

    if (aval != 121) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true extent of type = %d; should be %d\n", (int) aval, 121);
    }

    yaksa_type_free(tmptype);
    yaksa_type_free(inttype);
    yaksa_type_free(eviltype);

    return errs;
}

int int_with_negextent_test(void)
{
    int err, errs = 0, val;
    intptr_t lb, extent, aval, true_lb;
    char *typemapstring = 0;

    yaksa_type_t tmptype, eviltype;

    typemapstring = (char *) "{ 4*(BYTE,0) }";
    err = yaksa_type_create_contig(4, YAKSA_TYPE__BYTE, NULL, &tmptype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_contig of %s failed.\n", typemapstring);
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    typemapstring = (char *) "{ (LB,6),4*(BYTE,0),(UB,-3) }";
    err = yaksa_type_create_resized(tmptype, 6, -9, NULL, &eviltype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_resized of %s failed.\n", typemapstring);
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_get_size(eviltype, (uintptr_t *) (void *) &val);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_size failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (val != 4) {
        errs++;
        if (verbose)
            fprintf(stderr, "  size of type = %d; should be %d\n", val, 4);
    }

    err = yaksa_type_get_extent(eviltype, &lb, &extent);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (lb != 6) {
        errs++;
        if (verbose)
            fprintf(stderr, "  lb of type = %d; should be %d\n", (int) aval, 6);
    }

    if (extent != -9) {
        errs++;
        if (verbose)
            fprintf(stderr, "  extent of type = %d; should be %d\n", (int) extent, -9);
    }

    err = yaksa_type_get_true_extent(eviltype, &true_lb, &aval);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_true_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (true_lb != 0) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true_lb of type = %d; should be %d\n", (int) true_lb, 0);
    }

    if (aval != 4) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true extent of type = %d; should be %d\n", (int) aval, 4);
    }

    yaksa_type_free(tmptype);
    yaksa_type_free(eviltype);

    return errs;
}

int vector_blklen_stride_negextent_of_int_with_lb_ub_test(void)
{
    int err, errs = 0, val;
    intptr_t lb, extent, true_lb, aval;
    yaksa_type_t tmptype, inttype, eviltype;
    char *typemapstring = 0;

    /* build same type as in int_with_lb_ub_test() */
    typemapstring = (char *) "{ 4*(BYTE,0) }";
    err = yaksa_type_create_contig(4, YAKSA_TYPE__BYTE, NULL, &tmptype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_contig of %s failed.\n", typemapstring);
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    typemapstring = (char *) "{ (LB,6),4*(BYTE,0),(UB,-3) }";
    err = yaksa_type_create_resized(tmptype, 6, -9, NULL, &inttype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_resized of %s failed.\n", typemapstring);
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_create_vector(3, 4, 5, inttype, NULL, &eviltype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_vector failed.\n");
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_get_size(eviltype, (uintptr_t *) (void *) &val);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_size failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (val != 48) {
        errs++;
        if (verbose)
            fprintf(stderr, "  size of type = %d; should be %d\n", val, 48);
    }

    err = yaksa_type_get_extent(eviltype, &lb, &extent);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (lb != -111) {
        errs++;
        if (verbose)
            fprintf(stderr, "  lb of type = %d; should be %d\n", (int) aval, -111);
    }

    if (extent != 108) {
        errs++;
        if (verbose)
            fprintf(stderr, "  extent of type = %d; should be %d\n", (int) extent, 108);
    }

    err = yaksa_type_get_true_extent(eviltype, &true_lb, &aval);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_true_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (true_lb != -117) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true_lb of type = %d; should be %d\n", (int) true_lb, -117);
    }

    if (aval != 121) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true extent of type = %d; should be %d\n", (int) aval, 121);
    }

    yaksa_type_free(tmptype);
    yaksa_type_free(inttype);
    yaksa_type_free(eviltype);

    return errs;
}

int vector_blklen_negstride_negextent_of_int_with_lb_ub_test(void)
{
    int err, errs = 0, val;
    intptr_t extent, lb, aval, true_lb;

    yaksa_type_t tmptype, inttype, eviltype;

    /* build same type as in int_with_lb_ub_test() */
    err = yaksa_type_create_contig(4, YAKSA_TYPE__BYTE, NULL, &tmptype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_contig failed.\n");
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_create_resized(tmptype, 6, -9, NULL, &inttype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_resized failed.\n");
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_create_vector(3, 4, -5, inttype, NULL, &eviltype);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_create_vector failed.\n");
        if (verbose)
            TestPrintError(err);
        /* no point in continuing */
        return errs;
    }

    err = yaksa_type_get_size(eviltype, (uintptr_t *) (void *) &val);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_size failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (val != 48) {
        errs++;
        if (verbose)
            fprintf(stderr, "  size of type = %d; should be %d\n", val, 48);
    }

    err = yaksa_type_get_extent(eviltype, &lb, &extent);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (lb != -21) {
        errs++;
        if (verbose)
            fprintf(stderr, "  lb of type = %ld; should be %d\n", (long) aval, -21);
    }

    if (extent != 108) {
        errs++;
        if (verbose)
            fprintf(stderr, "  extent of type = %ld; should be %d\n", (long) extent, 108);
    }

    err = yaksa_type_get_true_extent(eviltype, &true_lb, &aval);
    if (err != YAKSA_SUCCESS) {
        errs++;
        if (verbose)
            fprintf(stderr, "  yaksa_type_get_true_extent failed.\n");
        if (verbose)
            TestPrintError(err);
    }

    if (true_lb != -27) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true_lb of type = %ld; should be %d\n", (long) true_lb, -27);
    }

    if (aval != 121) {
        errs++;
        if (verbose)
            fprintf(stderr, "  true extent of type = %ld; should be %d\n", (long) aval, 121);
    }

    yaksa_type_free(tmptype);
    yaksa_type_free(inttype);
    yaksa_type_free(eviltype);

    return errs;
}
