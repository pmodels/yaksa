/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

#define INIT_BUILTIN(c_type, TYPE, rc, fn_fail)                         \
    do {                                                                \
        yaksi_type_s *tmp_type_;                                        \
        /* builtin datatypes do not need a special allocation; we can   \
         * simply fetch them */                                         \
        rc = yaksi_type_get(YAKSA_TYPE__##TYPE, &tmp_type_);            \
        YAKSU_ERR_CHECK(rc, fn_fail);                                   \
                                                                        \
        tmp_type_->kind = YAKSI_TYPE_KIND__BUILTIN;                     \
        tmp_type_->tree_depth = 0;                                      \
                                                                        \
        tmp_type_->size = sizeof(c_type);                               \
        struct {                                                        \
            c_type x;                                                   \
            char y;                                                     \
        } z;                                                            \
        tmp_type_->alignment = sizeof(z) - sizeof(c_type);              \
        tmp_type_->extent = sizeof(c_type);                             \
        tmp_type_->lb = 0;                                              \
        tmp_type_->ub = sizeof(c_type);                                 \
        tmp_type_->true_lb = 0;                                         \
        tmp_type_->true_ub = sizeof(c_type);                            \
                                                                        \
        tmp_type_->is_contig = true;                                    \
        tmp_type_->num_contig = 1;                                      \
                                                                        \
        yaksur_type_create_hook(tmp_type_);                             \
    } while (0)

#define INIT_BUILTIN_PAIRTYPE(c_type1, c_type2, c_type, TYPE, rc, fn_fail) \
    do {                                                                \
        yaksi_type_s *tmp_type_;                                        \
        /* builtin datatypes do not need a special allocation; we can   \
         * simply fetch them */                                         \
        rc = yaksi_type_get(YAKSA_TYPE__##TYPE, &tmp_type_);            \
        YAKSU_ERR_CHECK(rc, fn_fail);                                   \
                                                                        \
        c_type z;                                                       \
        bool element_is_contig;                                         \
        if ((char *) &z.y - (char *) &z == sizeof(c_type1))             \
            element_is_contig = true;                                   \
        else                                                            \
            element_is_contig = false;                                  \
                                                                        \
        tmp_type_->kind = YAKSI_TYPE_KIND__BUILTIN;                     \
        tmp_type_->tree_depth = 0;                                      \
                                                                        \
        tmp_type_->size = sizeof(c_type1) + sizeof(c_type2);            \
        struct {                                                        \
            c_type1 x;                                                  \
            char y;                                                     \
        } z1;                                                           \
        struct {                                                        \
            c_type2 x;                                                  \
            char y;                                                     \
        } z2;                                                           \
        tmp_type_->alignment = YAKSU_MAX(sizeof(z1) - sizeof(c_type1), sizeof(z2) - sizeof(c_type2)); \
        tmp_type_->extent = sizeof(c_type);                             \
        tmp_type_->lb = 0;                                              \
        tmp_type_->ub = sizeof(c_type);                                 \
        tmp_type_->true_lb = 0;                                         \
        if (element_is_contig)                                          \
            tmp_type_->true_ub = tmp_type_->size;                       \
        else                                                            \
            tmp_type_->true_ub = sizeof(c_type);                        \
                                                                        \
        if (tmp_type_->size == tmp_type_->extent) {                     \
            tmp_type_->is_contig = true;                                \
            tmp_type_->num_contig = 1;                                  \
        } else {                                                        \
            tmp_type_->is_contig = false;                               \
            tmp_type_->num_contig = 2;                                  \
        }                                                               \
                                                                        \
        yaksur_type_create_hook(tmp_type_);                             \
    } while (0)

yaksi_global_s yaksi_global = { 0 };
yaksa_init_attr_t YAKSA_INIT_ATTR__DEFAULT = { 0 };

#define CHUNK_SIZE (1024)

int yaksa_init(yaksa_init_attr_t attr)
{
    int rc = YAKSA_SUCCESS;

    /* initialize the type pool */
    rc = yaksu_pool_alloc(sizeof(yaksi_type_s), CHUNK_SIZE, UINTPTR_MAX, malloc, free,
                          &yaksi_global.type_pool);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* these are the first set of allocations for our builtin
     * datatypes, so the indices should match that of the datatype
     * themselves */
    for (int i = 0; i < YAKSI_TYPE__LAST; i++) {
        struct yaksi_type_s *type;

        rc = yaksi_type_alloc(&type);
        YAKSU_ERR_CHECK(rc, fn_fail);

        assert(type->id == i);
    }

    /* initialize the request pool */
    rc = yaksu_pool_alloc(sizeof(yaksi_request_s), CHUNK_SIZE, UINTPTR_MAX, malloc, free,
                          &yaksi_global.request_pool);
    YAKSU_ERR_CHECK(rc, fn_fail);


    /* initialize the backend */
    rc = yaksur_init_hook();
    YAKSU_ERR_CHECK(rc, fn_fail);


    /* setup builtin datatypes */
    INIT_BUILTIN(char, CHAR, rc, fn_fail);
    INIT_BUILTIN(unsigned char, UNSIGNED_CHAR, rc, fn_fail);
    INIT_BUILTIN(wchar_t, WCHAR_T, rc, fn_fail);

    INIT_BUILTIN(int, INT, rc, fn_fail);
    INIT_BUILTIN(unsigned, UNSIGNED, rc, fn_fail);
    INIT_BUILTIN(short, SHORT, rc, fn_fail);
    INIT_BUILTIN(unsigned short, UNSIGNED_SHORT, rc, fn_fail);
    INIT_BUILTIN(long, LONG, rc, fn_fail);
    INIT_BUILTIN(unsigned long, UNSIGNED_LONG, rc, fn_fail);
    INIT_BUILTIN(long long, LONG_LONG, rc, fn_fail);
    INIT_BUILTIN(unsigned long long, UNSIGNED_LONG_LONG, rc, fn_fail);
    INIT_BUILTIN(int8_t, INT8_T, rc, fn_fail);
    INIT_BUILTIN(int16_t, INT16_T, rc, fn_fail);
    INIT_BUILTIN(int32_t, INT32_T, rc, fn_fail);
    INIT_BUILTIN(int64_t, INT64_T, rc, fn_fail);
    INIT_BUILTIN(uint8_t, UINT8_T, rc, fn_fail);
    INIT_BUILTIN(uint16_t, UINT16_T, rc, fn_fail);
    INIT_BUILTIN(uint32_t, UINT32_T, rc, fn_fail);
    INIT_BUILTIN(uint64_t, UINT64_T, rc, fn_fail);

    INIT_BUILTIN(float, FLOAT, rc, fn_fail);
    INIT_BUILTIN(double, DOUBLE, rc, fn_fail);
    INIT_BUILTIN(long double, LONG_DOUBLE, rc, fn_fail);
    INIT_BUILTIN(float, C_COMPLEX, rc, fn_fail);
    INIT_BUILTIN(double, C_DOUBLE_COMPLEX, rc, fn_fail);
    INIT_BUILTIN(long double, C_LONG_DOUBLE_COMPLEX, rc, fn_fail);

    INIT_BUILTIN_PAIRTYPE(float, int, yaksi_float_int_s, FLOAT_INT, rc, fn_fail);
    INIT_BUILTIN_PAIRTYPE(double, int, yaksi_double_int_s, DOUBLE_INT, rc, fn_fail);
    INIT_BUILTIN_PAIRTYPE(long, int, yaksi_long_int_s, LONG_INT, rc, fn_fail);
    /* *INDENT-OFF* */
    INIT_BUILTIN_PAIRTYPE(int, int, yaksi_2int_s, 2INT, rc, fn_fail);
    /* *INDENT-ON* */
    INIT_BUILTIN_PAIRTYPE(short, int, yaksi_short_int_s, SHORT_INT, rc, fn_fail);
    INIT_BUILTIN_PAIRTYPE(long double, int, yaksi_long_double_int_s, LONG_DOUBLE_INT, rc, fn_fail);

    INIT_BUILTIN(uint8_t, BYTE, rc, fn_fail);


    /* done */
    yaksi_global.is_initialized = 1;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksa_finalize(void)
{
    int rc = YAKSA_SUCCESS;

    /* finalize the backend */
    rc = yaksur_finalize_hook();
    YAKSU_ERR_CHECK(rc, fn_fail);


    /* free the builtin datatypes */
    for (int i = 0; i < YAKSI_TYPE__LAST; i++) {
        yaksi_type_s *type;

        rc = yaksi_type_get(i, &type);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = yaksi_type_free(type);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    rc = yaksu_pool_free(yaksi_global.type_pool);
    YAKSU_ERR_CHECK(rc, fn_fail);


    /* free the builtin requests */
    for (int i = 0; i < YAKSI_REQUEST__LAST; i++) {
        yaksi_request_s *request;

        rc = yaksi_request_get(i, &request);
        YAKSU_ERR_CHECK(rc, fn_fail);

        rc = yaksi_request_free(request);
        YAKSU_ERR_CHECK(rc, fn_fail);
    }

    rc = yaksu_pool_free(yaksi_global.request_pool);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
