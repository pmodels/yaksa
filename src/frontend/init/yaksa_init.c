/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <limits.h>

#define ALLOC_TYPE_HANDLE(tmp_type_, TYPE, rc, fn_fail)                 \
    do {                                                                \
        tmp_type_ = (yaksi_type_s *) malloc(sizeof(yaksi_type_s));      \
        YAKSU_ERR_CHKANDJUMP(!tmp_type_, rc, YAKSA_ERR__OUT_OF_MEM, fn_fail); \
        yaksu_atomic_store(&tmp_type_->refcount, 1);                    \
                                                                        \
        yaksi_type_user_handle_s *user_handle;                          \
        user_handle = (yaksi_type_user_handle_s *) malloc(sizeof(yaksi_type_user_handle_s)); \
        YAKSU_ERR_CHKANDJUMP(!user_handle, rc, YAKSA_ERR__OUT_OF_MEM, fn_fail); \
                                                                        \
        rc = yaksi_type_handle_alloc(tmp_type_, &user_handle->id);      \
        YAKSU_ERR_CHECK(rc, fn_fail);                                   \
                                                                        \
        assert(user_handle->id == YAKSA_TYPE__##TYPE);                  \
    } while (0)

#define INIT_BUILTIN_TYPE(c_type, TYPE, rc, fn_fail)            \
    do {                                                        \
        yaksi_type_s *tmp_type_;                                \
        ALLOC_TYPE_HANDLE(tmp_type_, TYPE, rc, fn_fail);        \
                                                                \
        tmp_type_->u.builtin.handle = YAKSA_TYPE__##TYPE;       \
        tmp_type_->kind = YAKSI_TYPE_KIND__BUILTIN;             \
        tmp_type_->tree_depth = 0;                              \
                                                                \
        tmp_type_->size = sizeof(c_type);                       \
        struct {                                                \
            c_type x;                                           \
            char y;                                             \
        } z;                                                    \
        tmp_type_->alignment = sizeof(z) - sizeof(c_type);      \
        tmp_type_->extent = sizeof(c_type);                     \
        tmp_type_->lb = 0;                                      \
        tmp_type_->ub = sizeof(c_type);                         \
        tmp_type_->true_lb = 0;                                 \
        tmp_type_->true_ub = sizeof(c_type);                    \
                                                                \
        tmp_type_->is_contig = true;                            \
        tmp_type_->num_contig = 1;                              \
                                                                \
        yaksur_type_create_hook(tmp_type_);                     \
    } while (0)

#define INIT_BUILTIN_PAIRTYPE(c_type1, c_type2, c_type, TYPE, rc, fn_fail) \
    do {                                                                \
        yaksi_type_s *tmp_type_;                                        \
        ALLOC_TYPE_HANDLE(tmp_type_, TYPE, rc, fn_fail);                \
                                                                        \
        c_type z;                                                       \
        bool element_is_contig;                                         \
        if ((char *) &z.y - (char *) &z == sizeof(c_type1))             \
            element_is_contig = true;                                   \
        else                                                            \
            element_is_contig = false;                                  \
                                                                        \
        tmp_type_->u.builtin.handle = YAKSA_TYPE__##TYPE;               \
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
        } else {                                                        \
            tmp_type_->is_contig = false;                               \
        }                                                               \
        tmp_type_->num_contig = 1 + !element_is_contig;                 \
                                                                        \
        yaksur_type_create_hook(tmp_type_);                             \
    } while (0)

#define FINALIZE_BUILTIN_TYPE(TYPE, rc, fn_fail)                        \
    do {                                                                \
        uint32_t id = (uint32_t) YAKSA_TYPE__##TYPE;                    \
        yaksi_type_s *tmp_type_;                                        \
                                                                        \
        rc = yaksi_type_handle_dealloc(id, &tmp_type_);                 \
        YAKSU_ERR_CHECK(rc, fn_fail);                                   \
                                                                        \
        rc = yaksi_type_free(tmp_type_);                                \
        YAKSU_ERR_CHECK(rc, fn_fail);                                   \
    } while (0)

yaksi_global_s yaksi_global = { 0 };
yaksa_init_attr_t YAKSA_INIT_ATTR__DEFAULT = { 0 };

#define CHUNK_SIZE (1024)

int yaksa_init(yaksa_init_attr_t attr)
{
    int rc = YAKSA_SUCCESS;

    /*************************************************************/
    /* initialize the backend */
    /*************************************************************/
    rc = yaksur_init_hook();
    YAKSU_ERR_CHECK(rc, fn_fail);


    /*************************************************************/
    /* setup builtin datatypes */
    /*************************************************************/
    rc = yaksu_handle_pool_alloc(&yaksi_global.type_handle_pool);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* special handling for the NULL type */
    yaksi_type_s *null_type;
    ALLOC_TYPE_HANDLE(null_type, NULL, rc, fn_fail);

    null_type->u.builtin.handle = YAKSA_TYPE__NULL;
    null_type->kind = YAKSI_TYPE_KIND__BUILTIN;
    null_type->tree_depth = 0;
    null_type->size = 0;
    null_type->alignment = 1;
    null_type->extent = 0;
    null_type->lb = 0;
    null_type->ub = 0;
    null_type->true_lb = 0;
    null_type->true_ub = 0;

    null_type->is_contig = true;
    null_type->num_contig = 0;
    yaksur_type_create_hook(null_type);

    INIT_BUILTIN_TYPE(uint8_t, BYTE, rc, fn_fail);

    INIT_BUILTIN_TYPE(char, CHAR, rc, fn_fail);
    INIT_BUILTIN_TYPE(unsigned char, UNSIGNED_CHAR, rc, fn_fail);
    INIT_BUILTIN_TYPE(wchar_t, WCHAR_T, rc, fn_fail);

    INIT_BUILTIN_TYPE(int, INT, rc, fn_fail);
    INIT_BUILTIN_TYPE(unsigned, UNSIGNED, rc, fn_fail);
    INIT_BUILTIN_TYPE(short, SHORT, rc, fn_fail);
    INIT_BUILTIN_TYPE(unsigned short, UNSIGNED_SHORT, rc, fn_fail);
    INIT_BUILTIN_TYPE(long, LONG, rc, fn_fail);
    INIT_BUILTIN_TYPE(unsigned long, UNSIGNED_LONG, rc, fn_fail);
    INIT_BUILTIN_TYPE(long long, LONG_LONG, rc, fn_fail);
    INIT_BUILTIN_TYPE(unsigned long long, UNSIGNED_LONG_LONG, rc, fn_fail);
    INIT_BUILTIN_TYPE(int8_t, INT8_T, rc, fn_fail);
    INIT_BUILTIN_TYPE(int16_t, INT16_T, rc, fn_fail);
    INIT_BUILTIN_TYPE(int32_t, INT32_T, rc, fn_fail);
    INIT_BUILTIN_TYPE(int64_t, INT64_T, rc, fn_fail);
    INIT_BUILTIN_TYPE(uint8_t, UINT8_T, rc, fn_fail);
    INIT_BUILTIN_TYPE(uint16_t, UINT16_T, rc, fn_fail);
    INIT_BUILTIN_TYPE(uint32_t, UINT32_T, rc, fn_fail);
    INIT_BUILTIN_TYPE(uint64_t, UINT64_T, rc, fn_fail);

    INIT_BUILTIN_TYPE(float, FLOAT, rc, fn_fail);
    INIT_BUILTIN_TYPE(double, DOUBLE, rc, fn_fail);
    INIT_BUILTIN_TYPE(long double, LONG_DOUBLE, rc, fn_fail);

    INIT_BUILTIN_PAIRTYPE(float, float, yaksi_c_complex_s, C_COMPLEX, rc, fn_fail);
    INIT_BUILTIN_PAIRTYPE(double, double, yaksi_c_double_complex_s, C_DOUBLE_COMPLEX, rc, fn_fail);
    INIT_BUILTIN_PAIRTYPE(long double, long double, yaksi_c_long_double_complex_s,
                          C_LONG_DOUBLE_COMPLEX, rc, fn_fail);
    INIT_BUILTIN_PAIRTYPE(float, int, yaksi_float_int_s, FLOAT_INT, rc, fn_fail);
    INIT_BUILTIN_PAIRTYPE(double, int, yaksi_double_int_s, DOUBLE_INT, rc, fn_fail);
    INIT_BUILTIN_PAIRTYPE(long, int, yaksi_long_int_s, LONG_INT, rc, fn_fail);
    /* *INDENT-OFF* */
    INIT_BUILTIN_PAIRTYPE(int, int, yaksi_2int_s, 2INT, rc, fn_fail);
    /* *INDENT-ON* */
    INIT_BUILTIN_PAIRTYPE(short, int, yaksi_short_int_s, SHORT_INT, rc, fn_fail);
    INIT_BUILTIN_PAIRTYPE(long double, int, yaksi_long_double_int_s, LONG_DOUBLE_INT, rc, fn_fail);


    /*************************************************************/
    /* setup builtin requests */
    /*************************************************************/
    rc = yaksu_handle_pool_alloc(&yaksi_global.request_handle_pool);
    YAKSU_ERR_CHECK(rc, fn_fail);

    /* allocate the NULL request */
    struct yaksi_request_s *request;
    rc = yaksi_request_create(&request);
    YAKSU_ERR_CHECK(rc, fn_fail);

    assert(request->id == YAKSA_REQUEST__NULL);


    /*************************************************************/
    /* final processing */
    /*************************************************************/
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
    FINALIZE_BUILTIN_TYPE(NULL, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(BYTE, rc, fn_fail);

    FINALIZE_BUILTIN_TYPE(CHAR, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(UNSIGNED_CHAR, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(WCHAR_T, rc, fn_fail);

    FINALIZE_BUILTIN_TYPE(INT, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(UNSIGNED, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(SHORT, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(UNSIGNED_SHORT, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(LONG, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(UNSIGNED_LONG, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(LONG_LONG, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(UNSIGNED_LONG_LONG, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(INT8_T, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(INT16_T, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(INT32_T, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(INT64_T, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(UINT8_T, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(UINT16_T, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(UINT32_T, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(UINT64_T, rc, fn_fail);

    FINALIZE_BUILTIN_TYPE(FLOAT, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(DOUBLE, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(LONG_DOUBLE, rc, fn_fail);

    FINALIZE_BUILTIN_TYPE(C_COMPLEX, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(C_DOUBLE_COMPLEX, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(C_LONG_DOUBLE_COMPLEX, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(FLOAT_INT, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(DOUBLE_INT, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(LONG_INT, rc, fn_fail);
    /* *INDENT-OFF* */
    FINALIZE_BUILTIN_TYPE(2INT, rc, fn_fail);
    /* *INDENT-ON* */
    FINALIZE_BUILTIN_TYPE(SHORT_INT, rc, fn_fail);
    FINALIZE_BUILTIN_TYPE(LONG_DOUBLE_INT, rc, fn_fail);

    rc = yaksu_handle_pool_free(yaksi_global.type_handle_pool);
    YAKSU_ERR_CHECK(rc, fn_fail);


    /* free the NULL request */
    uint32_t id;
    id = (uint32_t) YAKSA_REQUEST__NULL;
    yaksi_request_s *request;

    rc = yaksi_request_get(id, &request);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksi_request_free(request);
    YAKSU_ERR_CHECK(rc, fn_fail);

    rc = yaksu_handle_pool_free(yaksi_global.request_handle_pool);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
