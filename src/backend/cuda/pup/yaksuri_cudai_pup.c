/*
* Copyright (C) by Argonne National Laboratory
*     See COPYRIGHT in top-level directory
*/

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "yaksi.h"
#include "yaksuri_cudai.h"
#include <stdlib.h>

#define THREAD_BLOCK_SIZE  (256)
#define MAX_GRIDSZ_X       ((1U << 31) - 1)
#define MAX_GRIDSZ_Y       (65535U)
#define MAX_GRIDSZ_Z       (65535U)

#define MAX_IOV_LENGTH (16384)

static int get_thread_block_dims(uintptr_t count, yaksi_type_s * type, unsigned int *n_threads,
                                 unsigned int *n_blocks_x, unsigned int *n_blocks_y,
                                 unsigned int *n_blocks_z)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_type_s *cuda_type = (yaksuri_cudai_type_s *) type->backend.cuda.priv;

    *n_threads = THREAD_BLOCK_SIZE;
    uintptr_t n_blocks = count * cuda_type->num_elements / THREAD_BLOCK_SIZE;
    n_blocks += ! !(count * cuda_type->num_elements % THREAD_BLOCK_SIZE);

    if (n_blocks <= MAX_GRIDSZ_X) {
        *n_blocks_x = (unsigned int) n_blocks;
        *n_blocks_y = 1;
        *n_blocks_z = 1;
    } else if (n_blocks <= MAX_GRIDSZ_X * MAX_GRIDSZ_Y) {
        *n_blocks_x = (unsigned int) (YAKSU_CEIL(n_blocks, MAX_GRIDSZ_Y));
        *n_blocks_y = (unsigned int) (YAKSU_CEIL(n_blocks, (*n_blocks_x)));
        *n_blocks_z = 1;
    } else {
        uintptr_t n_blocks_xy = YAKSU_CEIL(n_blocks, MAX_GRIDSZ_Z);
        *n_blocks_x = (unsigned int) (YAKSU_CEIL(n_blocks_xy, MAX_GRIDSZ_Y));
        *n_blocks_y = (unsigned int) (YAKSU_CEIL(n_blocks_xy, (*n_blocks_x)));
        *n_blocks_z =
            (unsigned int) (YAKSU_CEIL(n_blocks, (uintptr_t) (*n_blocks_x) * (*n_blocks_y)));
    }

    return rc;
}

int yaksuri_cudai_pup_is_supported(yaksi_type_s * type, bool * is_supported)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_type_s *cuda_type = (yaksuri_cudai_type_s *) type->backend.cuda.priv;

    if (type->is_contig || cuda_type->pack)
        *is_supported = true;
    else
        *is_supported = false;

    return rc;
}

int yaksuri_cudai_ipack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                        yaksi_info_s * info, void **event, void *gpu_tmpbuf, int target)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_type_s *cuda_type = (yaksuri_cudai_type_s *) type->backend.cuda.priv;
    cudaError_t cerr;

    uintptr_t iov_pack_threshold = YAKSURI_CUDAI_INFO__DEFAULT_IOV_PUP_THRESHOLD;
    if (info) {
        yaksuri_cudai_info_s *cuda_info = (yaksuri_cudai_info_s *) info->backend.cuda.priv;
        iov_pack_threshold = cuda_info->iov_pack_threshold;
    }

    struct cudaPointerAttributes outattr, inattr;

    cerr = cudaPointerGetAttributes(&inattr, (char *) inbuf + type->true_lb);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    cerr = cudaPointerGetAttributes(&outattr, outbuf);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    int cur_device;
    cerr = cudaGetDevice(&cur_device);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    cerr = cudaSetDevice(target);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    if (*event == NULL) {
        cerr = cudaEventCreate((cudaEvent_t *) event);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
    }

    /* shortcut for contiguous types */
    if (type->is_contig) {
        /* cuda performance is optimized when we synchronize on the
         * source buffer's GPU */
        cerr = cudaMemcpyAsync(outbuf, inbuf, count * type->size, cudaMemcpyDefault,
                               yaksuri_cudai_global.stream[target]);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
    } else if (type->size / type->num_contig >= iov_pack_threshold) {
        struct iovec *iov;
        uintptr_t actual_iov_len;

        if (type->num_contig * count <= MAX_IOV_LENGTH) {
            iov = (struct iovec *) malloc(type->num_contig * count * sizeof(struct iovec));

            rc = yaksi_iov(inbuf, count, type, 0, iov, MAX_IOV_LENGTH, &actual_iov_len);
            YAKSU_ERR_CHECK(rc, fn_fail);
            assert(actual_iov_len == type->num_contig * count);

            char *dbuf = (char *) outbuf;
            for (uintptr_t i = 0; i < actual_iov_len; i++) {
                cudaMemcpyAsync(dbuf, iov[i].iov_base, iov[i].iov_len, cudaMemcpyDefault,
                                yaksuri_cudai_global.stream[target]);
                dbuf += iov[i].iov_len;
            }

            free(iov);
        } else if (type->num_contig <= MAX_IOV_LENGTH) {
            iov = (struct iovec *) malloc(type->num_contig * sizeof(struct iovec));

            uintptr_t iov_offset = 0;
            char *dbuf = (char *) outbuf;
            const char *sbuf = (const char *) inbuf;
            for (uintptr_t i = 0; i < count; i++) {
                rc = yaksi_iov(sbuf, 1, type, iov_offset, iov, MAX_IOV_LENGTH, &actual_iov_len);
                YAKSU_ERR_CHECK(rc, fn_fail);
                assert(actual_iov_len == type->num_contig);

                for (uintptr_t j = 0; j < actual_iov_len; j++) {
                    cudaMemcpyAsync(dbuf, iov[j].iov_base, iov[j].iov_len, cudaMemcpyDefault,
                                    yaksuri_cudai_global.stream[target]);
                    dbuf += iov[j].iov_len;
                }

                sbuf += type->extent;
            }

            free(iov);
        } else {
            rc = YAKSA_ERR__NOT_SUPPORTED;
        }
    } else {
        rc = yaksuri_cudai_md_alloc(type);
        YAKSU_ERR_CHECK(rc, fn_fail);

        unsigned int n_threads;
        unsigned int n_blocks_x, n_blocks_y, n_blocks_z;
        rc = get_thread_block_dims(count, type, &n_threads, &n_blocks_x, &n_blocks_y, &n_blocks_z);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if ((inattr.type == cudaMemoryTypeManaged && outattr.type == cudaMemoryTypeManaged) ||
            (inattr.type == cudaMemoryTypeDevice && outattr.type == cudaMemoryTypeManaged) ||
            (inattr.type == cudaMemoryTypeDevice && outattr.type == cudaMemoryTypeDevice &&
             inattr.device == outattr.device)) {
            cuda_type->pack(inbuf, outbuf, count, cuda_type->md, n_threads, n_blocks_x, n_blocks_y,
                            n_blocks_z, target);
        } else if (inattr.type == cudaMemoryTypeManaged && outattr.type == cudaMemoryTypeDevice) {
            cuda_type->pack(inbuf, outbuf, count, cuda_type->md, n_threads, n_blocks_x, n_blocks_y,
                            n_blocks_z, target);
        } else if ((outattr.type == cudaMemoryTypeDevice && inattr.device != outattr.device) ||
                   (outattr.type == cudaMemoryTypeHost)) {
            assert(inattr.type == cudaMemoryTypeDevice);
            cuda_type->pack(inbuf, gpu_tmpbuf, count, cuda_type->md, n_threads, n_blocks_x,
                            n_blocks_y, n_blocks_z, target);
            cerr = cudaMemcpyAsync(outbuf, gpu_tmpbuf, count * type->size, cudaMemcpyDefault,
                                   yaksuri_cudai_global.stream[target]);
            YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
        } else {
            rc = YAKSA_ERR__INTERNAL;
            goto fn_fail;
        }
    }

    cerr = cudaEventRecord((cudaEvent_t) * event, yaksuri_cudai_global.stream[target]);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    cerr = cudaSetDevice(cur_device);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cudai_iunpack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                          yaksi_info_s * info, void **event, void *gpu_tmpbuf, int target)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_type_s *cuda_type = (yaksuri_cudai_type_s *) type->backend.cuda.priv;
    cudaError_t cerr;

    uintptr_t iov_unpack_threshold = YAKSURI_CUDAI_INFO__DEFAULT_IOV_PUP_THRESHOLD;
    if (info) {
        yaksuri_cudai_info_s *cuda_info = (yaksuri_cudai_info_s *) info->backend.cuda.priv;
        iov_unpack_threshold = cuda_info->iov_unpack_threshold;
    }

    struct cudaPointerAttributes outattr, inattr;

    cerr = cudaPointerGetAttributes(&inattr, inbuf);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    cerr = cudaPointerGetAttributes(&outattr, (char *) outbuf + type->true_lb);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    int cur_device;
    cerr = cudaGetDevice(&cur_device);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    cerr = cudaSetDevice(target);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    if (*event == NULL) {
        cerr = cudaEventCreate((cudaEvent_t *) event);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
    }

    /* shortcut for contiguous types */
    if (type->is_contig) {
        /* cuda performance is optimized when we synchronize on the
         * source buffer's GPU */
        cerr = cudaMemcpyAsync(outbuf, inbuf, count * type->size, cudaMemcpyDefault,
                               yaksuri_cudai_global.stream[target]);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
    } else if (type->size / type->num_contig >= iov_unpack_threshold) {
        struct iovec *iov;
        uintptr_t actual_iov_len;

        if (type->num_contig * count <= MAX_IOV_LENGTH) {
            iov = (struct iovec *) malloc(type->num_contig * count * sizeof(struct iovec));

            rc = yaksi_iov(outbuf, count, type, 0, iov, MAX_IOV_LENGTH, &actual_iov_len);
            YAKSU_ERR_CHECK(rc, fn_fail);
            assert(actual_iov_len == type->num_contig * count);

            const char *sbuf = (const char *) inbuf;
            for (uintptr_t i = 0; i < actual_iov_len; i++) {
                cudaMemcpyAsync(iov[i].iov_base, sbuf, iov[i].iov_len, cudaMemcpyDefault,
                                yaksuri_cudai_global.stream[target]);
                sbuf += iov[i].iov_len;
            }

            free(iov);
        } else if (type->num_contig <= MAX_IOV_LENGTH) {
            iov = (struct iovec *) malloc(type->num_contig * sizeof(struct iovec));

            uintptr_t iov_offset = 0;
            char *dbuf = (char *) outbuf;
            const char *sbuf = (const char *) inbuf;
            for (uintptr_t i = 0; i < count; i++) {
                rc = yaksi_iov(dbuf, 1, type, iov_offset, iov, MAX_IOV_LENGTH, &actual_iov_len);
                YAKSU_ERR_CHECK(rc, fn_fail);
                assert(actual_iov_len == type->num_contig);

                for (uintptr_t j = 0; j < actual_iov_len; j++) {
                    cudaMemcpyAsync(iov[j].iov_base, sbuf, iov[j].iov_len, cudaMemcpyDefault,
                                    yaksuri_cudai_global.stream[target]);
                    sbuf += iov[j].iov_len;
                }

                dbuf += type->extent;
            }

            free(iov);
        } else {
            rc = YAKSA_ERR__NOT_SUPPORTED;
        }
    } else {
        rc = yaksuri_cudai_md_alloc(type);
        YAKSU_ERR_CHECK(rc, fn_fail);

        unsigned int n_threads;
        unsigned int n_blocks_x, n_blocks_y, n_blocks_z;
        rc = get_thread_block_dims(count, type, &n_threads, &n_blocks_x, &n_blocks_y, &n_blocks_z);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if ((inattr.type == cudaMemoryTypeManaged && outattr.type == cudaMemoryTypeManaged) ||
            (inattr.type == cudaMemoryTypeManaged && outattr.type == cudaMemoryTypeDevice) ||
            (inattr.type == cudaMemoryTypeDevice && outattr.type == cudaMemoryTypeDevice &&
             inattr.device == outattr.device)) {
            cuda_type->unpack(inbuf, outbuf, count, cuda_type->md, n_threads, n_blocks_x,
                              n_blocks_y, n_blocks_z, target);
        } else if (inattr.type == cudaMemoryTypeDevice && outattr.type == cudaMemoryTypeManaged) {
            cuda_type->unpack(inbuf, outbuf, count, cuda_type->md, n_threads, n_blocks_x,
                              n_blocks_y, n_blocks_z, target);
        } else if ((inattr.type == cudaMemoryTypeDevice && inattr.device != outattr.device) ||
                   (inattr.type == cudaMemoryTypeHost)) {
            assert(outattr.type == cudaMemoryTypeDevice);

            cerr = cudaMemcpyAsync(gpu_tmpbuf, inbuf, count * type->size, cudaMemcpyDefault,
                                   yaksuri_cudai_global.stream[target]);
            YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

            cuda_type->unpack(gpu_tmpbuf, outbuf, count, cuda_type->md, n_threads, n_blocks_x,
                              n_blocks_y, n_blocks_z, target);
        } else {
            rc = YAKSA_ERR__INTERNAL;
            goto fn_fail;
        }
    }

    cerr = cudaEventRecord((cudaEvent_t) * event, yaksuri_cudai_global.stream[target]);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    cerr = cudaSetDevice(cur_device);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
