/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <yaksa.h>
#include <assert.h>
#include "matrix_util.h"

int main()
{
    int rc;
    int input_matrix[SIZE];
    int pack_buf[SIZE];
    int unpack_buf[SIZE];
    yaksa_type_t contig;

    yaksa_init(YAKSA_INIT_ATTR__DEFAULT);       /* before any yaksa API is called the library
                                                 * must be initialized */

    init_matrix(input_matrix, ROWS, COLS);
    set_matrix(pack_buf, ROWS, COLS, 0);
    set_matrix(unpack_buf, ROWS, COLS, 0);

    rc = yaksa_create_contig(SIZE, YAKSA_TYPE__INT, &contig);
    assert(rc == YAKSA_SUCCESS);

    /* pack */
    yaksa_request_t request;
    uintptr_t actual_pack_bytes;
    rc = yaksa_ipack(input_matrix, 1, contig, 0, pack_buf, SIZE * sizeof(int), &actual_pack_bytes,
                     &request);
    assert(rc == YAKSA_SUCCESS);
    rc = yaksa_request_wait(request);
    assert(rc == YAKSA_SUCCESS);

    /* unpack */
    rc = yaksa_iunpack(pack_buf, SIZE * sizeof(int), unpack_buf, 1, contig, 0, &request);
    assert(rc == YAKSA_SUCCESS);
    rc = yaksa_request_wait(request);
    assert(rc == YAKSA_SUCCESS);

    print_matrix(input_matrix, ROWS, COLS, "input_matrix=");
    print_matrix(pack_buf, ROWS, COLS, "pack_buf=");
    print_matrix(unpack_buf, ROWS, COLS, "unpack_buf=");

    yaksa_free(contig);
    yaksa_finalize();
    return 0;
}
