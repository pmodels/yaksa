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
    yaksa_type_t vector;
    yaksa_type_t vector_resized;
    yaksa_type_t transpose;

    yaksa_init(YAKSA_INIT_ATTR__DEFAULT);

    init_matrix(input_matrix, ROWS, COLS);
    set_matrix(pack_buf, ROWS, COLS, 0);
    set_matrix(unpack_buf, ROWS, COLS, 0);

    rc = yaksa_create_vector(ROWS, 1, COLS, YAKSA_TYPE__INT, &vector);
    assert(rc == YAKSA_SUCCESS);
    rc = yaksa_create_resized(vector, 0, sizeof(int), &vector_resized);
    assert(rc == YAKSA_SUCCESS);
    rc = yaksa_create_contig(COLS, vector_resized, &transpose);
    assert(rc == YAKSA_SUCCESS);

    yaksa_request_t request;
    uintptr_t actual_pack_bytes;
    rc = yaksa_ipack(input_matrix, 1, transpose, 0, pack_buf, 256, &actual_pack_bytes, &request);
    assert(rc == YAKSA_SUCCESS);
    rc = yaksa_request_wait(request);
    assert(rc == YAKSA_SUCCESS);

    uintptr_t actual_unpack_bytes;
    rc = yaksa_iunpack(pack_buf, 256, unpack_buf, 1, transpose, 0, &actual_unpack_bytes, &request);
    assert(rc == YAKSA_SUCCESS);
    rc = yaksa_request_wait(request);
    assert(rc == YAKSA_SUCCESS);

    print_matrix(input_matrix, ROWS, COLS, "input_matrix=");
    print_matrix(pack_buf, ROWS, COLS, "pack_buf=");
    print_matrix(unpack_buf, ROWS, COLS, "unpack_buf=");

    yaksa_free(vector);
    yaksa_free(vector_resized);
    yaksa_free(transpose);
    yaksa_finalize();
    return 0;
}
