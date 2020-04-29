/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "yaksa_config.h"
#include "yaksa.h"
#include "dtpools.h"

#ifdef HAVE_CUDA
#include <cuda_runtime_api.h>
#endif /* HAVE_CUDA */

uintptr_t maxbufsize = 512 * 1024 * 1024;

enum {
    PACK_ORDER__UNSET,
    PACK_ORDER__NORMAL,
    PACK_ORDER__REVERSE,
    PACK_ORDER__RANDOM,
};

enum {
    OVERLAP__NONE,
    OVERLAP__REGULAR,
    OVERLAP__IRREGULAR,
};

#define MAX_DTP_BASESTRLEN (1024)

static int verbose = 0;

#define dprintf(...)                            \
    do {                                        \
        if (verbose)                            \
            printf(__VA_ARGS__);                \
    } while (0)

static void swap_segments(uintptr_t * starts, uintptr_t * lengths, int x, int y)
{
    uintptr_t tmp = starts[x];
    starts[x] = starts[y];
    starts[y] = tmp;

    tmp = lengths[x];
    lengths[x] = lengths[y];
    lengths[y] = tmp;
}

typedef enum {
    MEM_TYPE__UNSET,
    MEM_TYPE__UNREGISTERED_HOST,
    MEM_TYPE__REGISTERED_HOST,
    MEM_TYPE__DEVICE,
} mem_type_e;

#ifdef HAVE_CUDA
static int ndevices = -1;
#endif
static int device_id = 0;
static int device_stride = 0;

static void init_devices(void)
{
#ifdef HAVE_CUDA
    cudaGetDeviceCount(&ndevices);
    assert(ndevices != -1);
    cudaSetDevice(device_id);
#endif
}

static void alloc_mem(size_t size, mem_type_e type, void **hostbuf, void **devicebuf)
{
    if (type == MEM_TYPE__UNREGISTERED_HOST) {
        *devicebuf = malloc(size);
        if (hostbuf)
            *hostbuf = *devicebuf;
#ifdef HAVE_CUDA
    } else if (type == MEM_TYPE__REGISTERED_HOST) {
        cudaMallocHost(devicebuf, size);
        if (hostbuf)
            *hostbuf = *devicebuf;
    } else if (type == MEM_TYPE__DEVICE) {
        cudaSetDevice(device_id);
        cudaMalloc(devicebuf, size);
        if (hostbuf)
            cudaMallocHost(hostbuf, size);
        device_id += device_stride;
        device_id %= ndevices;
#endif
    } else {
        fprintf(stderr, "ERROR: unsupported memory type\n");
        exit(1);
    }
}

static void free_mem(mem_type_e type, void *hostbuf, void *devicebuf)
{
    if (type == MEM_TYPE__UNREGISTERED_HOST) {
        free(hostbuf);
#ifdef HAVE_CUDA
    } else if (type == MEM_TYPE__REGISTERED_HOST) {
        cudaFreeHost(devicebuf);
    } else if (type == MEM_TYPE__DEVICE) {
        cudaFree(devicebuf);
        if (hostbuf) {
            cudaFreeHost(hostbuf);
        }
#endif
    }
}

static void copy_content(const void *sbuf, void *dbuf, size_t size, mem_type_e type)
{
#ifdef HAVE_CUDA
    if (type == MEM_TYPE__DEVICE) {
        cudaMemcpy(dbuf, sbuf, size, cudaMemcpyDefault);
    }
#endif
}

int main(int argc, char **argv)
{
    DTP_pool_s dtp;
    DTP_obj_s sobj, dobj;
    int rc;
    char typestr[MAX_DTP_BASESTRLEN + 1] = { 0 };
    int basecount = -1;
    int seed = -1;
    int iters = -1;
    int segments = -1;
    int pack_order = PACK_ORDER__UNSET;
    int overlap = -1;
    mem_type_e sbuf_memtype = MEM_TYPE__UNREGISTERED_HOST;
    mem_type_e dbuf_memtype = MEM_TYPE__UNREGISTERED_HOST;
    mem_type_e tbuf_memtype = MEM_TYPE__UNREGISTERED_HOST;

    while (--argc && ++argv) {
        if (!strcmp(*argv, "-datatype")) {
            --argc;
            ++argv;
            strncpy(typestr, *argv, MAX_DTP_BASESTRLEN);
        } else if (!strcmp(*argv, "-count")) {
            --argc;
            ++argv;
            basecount = atoi(*argv);
        } else if (!strcmp(*argv, "-seed")) {
            --argc;
            ++argv;
            seed = atoi(*argv);
        } else if (!strcmp(*argv, "-iters")) {
            --argc;
            ++argv;
            iters = atoi(*argv);
        } else if (!strcmp(*argv, "-segments")) {
            --argc;
            ++argv;
            segments = atoi(*argv);
        } else if (!strcmp(*argv, "-ordering")) {
            --argc;
            ++argv;
            if (!strcmp(*argv, "normal"))
                pack_order = PACK_ORDER__NORMAL;
            else if (!strcmp(*argv, "reverse"))
                pack_order = PACK_ORDER__REVERSE;
            else if (!strcmp(*argv, "random"))
                pack_order = PACK_ORDER__RANDOM;
            else {
                fprintf(stderr, "unknown packing order %s\n", *argv);
                exit(1);
            }
        } else if (!strcmp(*argv, "-overlap")) {
            --argc;
            ++argv;
            if (!strcmp(*argv, "none"))
                overlap = OVERLAP__NONE;
            else if (!strcmp(*argv, "regular"))
                overlap = OVERLAP__REGULAR;
            else if (!strcmp(*argv, "irregular"))
                overlap = OVERLAP__IRREGULAR;
            else {
                fprintf(stderr, "unknown overlap type %s\n", *argv);
                exit(1);
            }
        } else if (!strcmp(*argv, "-sbuf-memtype")) {
            --argc;
            ++argv;
            if (!strcmp(*argv, "unreg-host"))
                sbuf_memtype = MEM_TYPE__UNREGISTERED_HOST;
            else if (!strcmp(*argv, "reg-host"))
                sbuf_memtype = MEM_TYPE__REGISTERED_HOST;
            else if (!strcmp(*argv, "device"))
                sbuf_memtype = MEM_TYPE__DEVICE;
            else {
                fprintf(stderr, "unknown buffer type %s\n", *argv);
                exit(1);
            }
        } else if (!strcmp(*argv, "-dbuf-memtype")) {
            --argc;
            ++argv;
            if (!strcmp(*argv, "unreg-host"))
                dbuf_memtype = MEM_TYPE__UNREGISTERED_HOST;
            else if (!strcmp(*argv, "reg-host"))
                dbuf_memtype = MEM_TYPE__REGISTERED_HOST;
            else if (!strcmp(*argv, "device"))
                dbuf_memtype = MEM_TYPE__DEVICE;
            else {
                fprintf(stderr, "unknown buffer type %s\n", *argv);
                exit(1);
            }
        } else if (!strcmp(*argv, "-tbuf-memtype")) {
            --argc;
            ++argv;
            if (!strcmp(*argv, "unreg-host"))
                tbuf_memtype = MEM_TYPE__UNREGISTERED_HOST;
            else if (!strcmp(*argv, "reg-host"))
                tbuf_memtype = MEM_TYPE__REGISTERED_HOST;
            else if (!strcmp(*argv, "device"))
                tbuf_memtype = MEM_TYPE__DEVICE;
            else {
                fprintf(stderr, "unknown buffer type %s\n", *argv);
                exit(1);
            }
        } else if (!strcmp(*argv, "-device-start-id")) {
            --argc;
            ++argv;
            device_id = atoi(*argv);
        } else if (!strcmp(*argv, "-device-stride")) {
            --argc;
            ++argv;
            device_stride = atoi(*argv);
        } else if (!strcmp(*argv, "-verbose")) {
            verbose = 1;
        } else {
            fprintf(stderr, "unknown argument %s\n", *argv);
            exit(1);
        }
    }
    if (strlen(typestr) == 0 || basecount <= 0 || seed < 0 || iters < 0 || segments < 0 ||
        pack_order == PACK_ORDER__UNSET || overlap < 0) {
        fprintf(stderr, "Usage: ./pack {options}\n");
        fprintf(stderr, "   -datatype    base datatype to use, e.g., int\n");
        fprintf(stderr, "   -count       number of base datatypes in the signature\n");
        fprintf(stderr, "   -seed        random seed (changes the datatypes generated)\n");
        fprintf(stderr, "   -iters       number of iterations\n");
        fprintf(stderr, "   -segments    number of segments to chop the packing into\n");
        fprintf(stderr, "   -ordering  packing order of segments (normal, reverse, random)\n");
        fprintf(stderr, "   -overlap     should packing overlap (none, regular, irregular)\n");
        fprintf(stderr, "   -sbuf-memtype memory type (unreg-host, reg-host, device)\n");
        fprintf(stderr, "   -dbuf-memtype memory type (unreg-host, reg-host, device)\n");
        fprintf(stderr, "   -tbuf-memtype memory type (unreg-host, reg-host, device)\n");
        fprintf(stderr, "   -device-start-id  ID of the device for the first allocation\n");
        fprintf(stderr, "   -device-stride    difference between consecutive device allocations\n");
        fprintf(stderr, "   -verbose     verbose output\n");
        exit(1);
    }

    yaksa_init(YAKSA_INIT_ATTR__DEFAULT);
    init_devices();

    rc = DTP_pool_create(typestr, basecount, seed, &dtp);
    assert(rc == DTP_SUCCESS);

    uintptr_t *segment_starts = (uintptr_t *) malloc(segments * sizeof(uintptr_t));
    uintptr_t *segment_lengths = (uintptr_t *) malloc(segments * sizeof(uintptr_t));

    for (int i = 0; i < iters; i++) {
        dprintf("==== iter %d ====\n", i);

        /* create the source object */
        rc = DTP_obj_create(dtp, &sobj, maxbufsize);
        assert(rc == DTP_SUCCESS);

        char *sbuf_h = NULL, *sbuf_d = NULL;
        alloc_mem(sobj.DTP_bufsize, sbuf_memtype, (void **) &sbuf_h, (void **) &sbuf_d);
        assert(sbuf_h);
        assert(sbuf_d);

        if (verbose) {
            char *desc;
            rc = DTP_obj_get_description(sobj, &desc);
            assert(rc == DTP_SUCCESS);
            dprintf("==> sbuf_h %p, sbuf_d %p, sobj (count: %zu):\n%s\n", sbuf_h, sbuf_d,
                    sobj.DTP_type_count, desc);
            free(desc);
        }

        rc = DTP_obj_buf_init(sobj, sbuf_h, 0, 1, basecount);
        assert(rc == DTP_SUCCESS);

        uintptr_t ssize;
        rc = yaksa_get_size(sobj.DTP_datatype, &ssize);
        assert(rc == YAKSA_SUCCESS);


        /* create the destination object */
        rc = DTP_obj_create(dtp, &dobj, maxbufsize);
        assert(rc == DTP_SUCCESS);

        char *dbuf_h, *dbuf_d;
        alloc_mem(dobj.DTP_bufsize, dbuf_memtype, (void **) &dbuf_h, (void **) &dbuf_d);
        assert(dbuf_h);
        assert(dbuf_d);

        if (verbose) {
            char *desc;
            rc = DTP_obj_get_description(dobj, &desc);
            assert(rc == DTP_SUCCESS);
            dprintf("==> dbuf_h %p, dbuf_d %p, dobj (count: %zu):\n%s\n", dbuf_h, dbuf_d,
                    dobj.DTP_type_count, desc);
            free(desc);
        }

        rc = DTP_obj_buf_init(dobj, dbuf_h, -1, -1, basecount);
        assert(rc == DTP_SUCCESS);

        uintptr_t dsize;
        rc = yaksa_get_size(dobj.DTP_datatype, &dsize);
        assert(rc == YAKSA_SUCCESS);


        /* the source and destination objects should have the same
         * signature */
        assert(ssize * sobj.DTP_type_count == dsize * dobj.DTP_type_count);


        /* pack from the source object to a temporary buffer and
         * unpack into the destination object */

        /* figure out the lengths and offsets of each segment */
        uintptr_t type_size;
        rc = yaksa_get_size(dtp.DTP_base_type, &type_size);
        assert(rc == YAKSA_SUCCESS);

        while (((ssize * sobj.DTP_type_count) / type_size) % segments)
            segments--;

        uintptr_t offset = 0;
        for (int j = 0; j < segments; j++) {
            segment_starts[j] = offset;

            uintptr_t eqlength = ssize * sobj.DTP_type_count / segments;

            /* make sure eqlength is a multiple of type_size */
            eqlength = (eqlength / type_size) * type_size;

            if (overlap == OVERLAP__NONE) {
                segment_lengths[j] = eqlength;
                offset += eqlength;
            } else if (overlap == OVERLAP__REGULAR) {
                if (offset + 2 * eqlength <= ssize * sobj.DTP_type_count)
                    segment_lengths[j] = 2 * eqlength;
                else
                    segment_lengths[j] = eqlength;
                offset += eqlength;
            } else {
                if (j == segments - 1) {
                    if (ssize * sobj.DTP_type_count > offset)
                        segment_lengths[j] = ssize * sobj.DTP_type_count - offset;
                    else
                        segment_lengths[j] = 0;
                    segment_lengths[j] += rand() % eqlength;
                } else {
                    segment_lengths[j] = rand() % (ssize * sobj.DTP_type_count - offset + eqlength);
                }

                offset += ((rand() % (segment_lengths[j] + 1)) / type_size) * type_size;
                if (offset > ssize * sobj.DTP_type_count)
                    offset = ssize * sobj.DTP_type_count;
            }
        }

        /* update the order in which we access the segments */
        if (pack_order == PACK_ORDER__NORMAL) {
            /* nothing to do */
        } else if (pack_order == PACK_ORDER__REVERSE) {
            for (int j = 0; j < segments / 2; j++) {
                swap_segments(segment_starts, segment_lengths, j, segments - j - 1);
            }
        } else if (pack_order == PACK_ORDER__RANDOM) {
            for (int j = 0; j < 1000; j++) {
                int x = rand() % segments;
                int y = rand() % segments;
                swap_segments(segment_starts, segment_lengths, x, y);
            }
        }

        /* the actual pack/unpack loop */
        copy_content(sbuf_h, sbuf_d, sobj.DTP_bufsize, sbuf_memtype);
        copy_content(dbuf_h, dbuf_d, dobj.DTP_bufsize, dbuf_memtype);

        void *tbuf;
        uintptr_t tbufsize = ssize * sobj.DTP_type_count;
        alloc_mem(tbufsize, tbuf_memtype, NULL, (void **) &tbuf);

        for (int j = 0; j < segments; j++) {
            uintptr_t actual_pack_bytes;
            yaksa_request_t request;

            rc = yaksa_ipack(sbuf_d + sobj.DTP_buf_offset, sobj.DTP_type_count, sobj.DTP_datatype,
                             segment_starts[j], tbuf, segment_lengths[j], &actual_pack_bytes,
                             &request);
            assert(rc == YAKSA_SUCCESS);
            assert(actual_pack_bytes <= segment_lengths[j]);

            rc = yaksa_request_wait(request);
            assert(rc == YAKSA_SUCCESS);

            rc = yaksa_iunpack(tbuf, actual_pack_bytes, dbuf_d + dobj.DTP_buf_offset,
                               dobj.DTP_type_count, dobj.DTP_datatype, segment_starts[j], &request);
            assert(rc == YAKSA_SUCCESS);

            rc = yaksa_request_wait(request);
            assert(rc == YAKSA_SUCCESS);
        }

        copy_content(dbuf_d, dbuf_h, dobj.DTP_bufsize, dbuf_memtype);
        rc = DTP_obj_buf_check(dobj, dbuf_h, 0, 1, basecount);
        assert(rc == DTP_SUCCESS);


        /* free allocated buffers and objects */
        free_mem(sbuf_memtype, sbuf_h, sbuf_d);
        free_mem(dbuf_memtype, dbuf_h, dbuf_d);
        free_mem(tbuf_memtype, NULL, tbuf);

        DTP_obj_free(sobj);
        DTP_obj_free(dobj);
    }

    free(segment_lengths);
    free(segment_starts);
    DTP_pool_free(dtp);

    yaksa_finalize();

    return 0;
}
