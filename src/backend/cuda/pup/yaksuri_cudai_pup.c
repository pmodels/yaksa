/*
* Copyright (C) by Argonne National Laboratory
*     See COPYRIGHT in top-level directory
*/

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "yaksi.h"
#include "yaksuri_cudai.h"

#define THREAD_BLOCK_SIZE  (256)
#define MAX_GRIDSZ_X       ((1ULL << 31) - 1)
#define MAX_GRIDSZ_Y       (65535)
#define MAX_GRIDSZ_Z       (65535)

static int get_thread_block_dims(uint64_t count, yaksi_type_s * type, int *n_threads,
                                 int *n_blocks_x, int *n_blocks_y, int *n_blocks_z)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_type_s *cuda_type = (yaksuri_cudai_type_s *) type->backend.cuda.priv;

    *n_threads = THREAD_BLOCK_SIZE;
    uint64_t n_blocks = count * cuda_type->num_elements / THREAD_BLOCK_SIZE;
    n_blocks += ! !(count * cuda_type->num_elements % THREAD_BLOCK_SIZE);

    if (n_blocks <= MAX_GRIDSZ_X) {
        *n_blocks_x = (int) n_blocks;
        *n_blocks_y = 1;
        *n_blocks_z = 1;
    } else if (n_blocks <= MAX_GRIDSZ_X * MAX_GRIDSZ_Y) {
        *n_blocks_x = YAKSU_CEIL(n_blocks, MAX_GRIDSZ_Y);
        *n_blocks_y = YAKSU_CEIL(n_blocks, (*n_blocks_x));
        *n_blocks_z = 1;
    } else {
        int n_blocks_xy = YAKSU_CEIL(n_blocks, MAX_GRIDSZ_Z);
        *n_blocks_x = YAKSU_CEIL(n_blocks_xy, MAX_GRIDSZ_Y);
        *n_blocks_y = YAKSU_CEIL(n_blocks_xy, (*n_blocks_x));
        *n_blocks_z = YAKSU_CEIL(n_blocks, (uintptr_t) (*n_blocks_x) * (*n_blocks_y));
    }

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cudai_pup_is_supported(yaksi_type_s * type, bool * is_supported)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_type_s *cuda_type = (yaksuri_cudai_type_s *) type->backend.cuda.priv;

    if (type->is_contig || cuda_type->pack)
        *is_supported = true;
    else
        *is_supported = false;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_cudai_ipack(const void *inbuf, void *outbuf, uintptr_t count, yaksi_type_s * type,
                        void *device_tmpbuf, void *interm_event, void **event)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_type_s *cuda_type = (yaksuri_cudai_type_s *) type->backend.cuda.priv;
    cudaError_t cerr;

    struct cudaPointerAttributes outattr, inattr;

    cerr = cudaPointerGetAttributes(&inattr, (char *) inbuf + type->true_lb);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    cerr = cudaPointerGetAttributes(&outattr, outbuf);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    int cur_device;
    cerr = cudaGetDevice(&cur_device);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    int target = -1;

    /* shortcut for contiguous types */
    if (type->is_contig) {
        /* cuda performance is optimized when we synchronize on the
         * source buffer's GPU */
        target = inattr.device;

        cerr = cudaSetDevice(target);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

        if (interm_event) {
            cerr = cudaStreamWaitEvent(yaksuri_cudai_global.stream[target],
                                       (cudaEvent_t) interm_event, 0);
            YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
        }

        cerr = cudaMemcpyAsync(outbuf, inbuf, count * type->size, cudaMemcpyDefault,
                               yaksuri_cudai_global.stream[target]);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
    } else {
        rc = yaksuri_cudai_md_alloc(type);
        YAKSU_ERR_CHECK(rc, fn_fail);

        int n_threads;
        int n_blocks_x, n_blocks_y, n_blocks_z;
        rc = get_thread_block_dims(count, type, &n_threads, &n_blocks_x, &n_blocks_y, &n_blocks_z);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if ((inattr.type == cudaMemoryTypeManaged && outattr.type == cudaMemoryTypeManaged) ||
            (inattr.type == cudaMemoryTypeDevice && outattr.type == cudaMemoryTypeManaged) ||
            (inattr.type == cudaMemoryTypeDevice && outattr.type == cudaMemoryTypeDevice &&
             inattr.device == outattr.device)) {
            target = inattr.device;
            cerr = cudaSetDevice(target);
            YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

            if (interm_event) {
                cerr = cudaStreamWaitEvent(yaksuri_cudai_global.stream[target],
                                           (cudaEvent_t) interm_event, 0);
                YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
            }

            cuda_type->pack(inbuf, outbuf, count, cuda_type->md, n_threads, n_blocks_x, n_blocks_y,
                            n_blocks_z, target);
        } else if (inattr.type == cudaMemoryTypeManaged && outattr.type == cudaMemoryTypeDevice) {
            target = outattr.device;
            cerr = cudaSetDevice(target);
            YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

            if (interm_event) {
                cerr = cudaStreamWaitEvent(yaksuri_cudai_global.stream[target],
                                           (cudaEvent_t) interm_event, 0);
                YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
            }

            cuda_type->pack(inbuf, outbuf, count, cuda_type->md, n_threads, n_blocks_x, n_blocks_y,
                            n_blocks_z, target);
        } else if ((outattr.type == cudaMemoryTypeDevice && inattr.device != outattr.device) ||
                   (outattr.type == cudaMemoryTypeHost)) {
            assert(inattr.type == cudaMemoryTypeDevice);

            target = inattr.device;
            cerr = cudaSetDevice(target);
            YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

            if (interm_event) {
                cerr = cudaStreamWaitEvent(yaksuri_cudai_global.stream[target],
                                           (cudaEvent_t) interm_event, 0);
                YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
            }

            cuda_type->pack(inbuf, device_tmpbuf, count, cuda_type->md, n_threads, n_blocks_x,
                            n_blocks_y, n_blocks_z, target);
            cerr = cudaMemcpyAsync(outbuf, device_tmpbuf, count * type->size, cudaMemcpyDefault,
                                   yaksuri_cudai_global.stream[target]);
            YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
        } else {
            rc = YAKSA_ERR__INTERNAL;
            goto fn_fail;
        }
    }

    if (*event == NULL) {
        cerr = cudaEventCreate((cudaEvent_t *) event);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
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
                          void *device_tmpbuf, void *interm_event, void **event)
{
    int rc = YAKSA_SUCCESS;
    yaksuri_cudai_type_s *cuda_type = (yaksuri_cudai_type_s *) type->backend.cuda.priv;
    cudaError_t cerr;

    struct cudaPointerAttributes outattr, inattr;

    cerr = cudaPointerGetAttributes(&inattr, inbuf);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    cerr = cudaPointerGetAttributes(&outattr, (char *) outbuf + type->true_lb);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    int cur_device;
    cerr = cudaGetDevice(&cur_device);
    YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

    int target = -1;

    /* shortcut for contiguous types */
    if (type->is_contig) {
        /* cuda performance is optimized when we synchronize on the
         * source buffer's GPU */
        target = inattr.device;

        cerr = cudaSetDevice(target);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

        if (interm_event) {
            cerr = cudaStreamWaitEvent(yaksuri_cudai_global.stream[target],
                                       (cudaEvent_t) interm_event, 0);
            YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
        }

        cerr = cudaMemcpyAsync(outbuf, inbuf, count * type->size, cudaMemcpyDefault,
                               yaksuri_cudai_global.stream[target]);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
    } else {
        rc = yaksuri_cudai_md_alloc(type);
        YAKSU_ERR_CHECK(rc, fn_fail);

        int n_threads;
        int n_blocks_x, n_blocks_y, n_blocks_z;
        rc = get_thread_block_dims(count, type, &n_threads, &n_blocks_x, &n_blocks_y, &n_blocks_z);
        YAKSU_ERR_CHECK(rc, fn_fail);

        if ((inattr.type == cudaMemoryTypeManaged && outattr.type == cudaMemoryTypeManaged) ||
            (inattr.type == cudaMemoryTypeManaged && outattr.type == cudaMemoryTypeDevice) ||
            (inattr.type == cudaMemoryTypeDevice && outattr.type == cudaMemoryTypeDevice &&
             inattr.device == outattr.device)) {
            target = outattr.device;
            cerr = cudaSetDevice(target);
            YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

            if (interm_event) {
                cerr = cudaStreamWaitEvent(yaksuri_cudai_global.stream[target],
                                           (cudaEvent_t) interm_event, 0);
                YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
            }

            cuda_type->unpack(inbuf, outbuf, count, cuda_type->md, n_threads, n_blocks_x,
                              n_blocks_y, n_blocks_z, target);
        } else if (inattr.type == cudaMemoryTypeDevice && outattr.type == cudaMemoryTypeManaged) {
            target = inattr.device;
            cerr = cudaSetDevice(target);
            YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

            if (interm_event) {
                cerr = cudaStreamWaitEvent(yaksuri_cudai_global.stream[target],
                                           (cudaEvent_t) interm_event, 0);
                YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
            }

            cuda_type->unpack(inbuf, outbuf, count, cuda_type->md, n_threads, n_blocks_x,
                              n_blocks_y, n_blocks_z, target);
        } else if ((inattr.type == cudaMemoryTypeDevice && inattr.device != outattr.device) ||
                   (inattr.type == cudaMemoryTypeHost)) {
            assert(outattr.type == cudaMemoryTypeDevice);

            target = outattr.device;
            cerr = cudaSetDevice(target);
            YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

            if (interm_event) {
                cerr = cudaStreamWaitEvent(yaksuri_cudai_global.stream[target],
                                           (cudaEvent_t) interm_event, 0);
                YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
            }

            cerr = cudaMemcpyAsync(device_tmpbuf, inbuf, count * type->size, cudaMemcpyDefault,
                                   yaksuri_cudai_global.stream[target]);
            YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);

            cuda_type->unpack(device_tmpbuf, outbuf, count, cuda_type->md, n_threads, n_blocks_x,
                              n_blocks_y, n_blocks_z, target);
        } else {
            rc = YAKSA_ERR__INTERNAL;
            goto fn_fail;
        }
    }

    if (*event == NULL) {
        cerr = cudaEventCreate((cudaEvent_t *) event);
        YAKSURI_CUDAI_CUDA_ERR_CHKANDJUMP(cerr, rc, fn_fail);
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
