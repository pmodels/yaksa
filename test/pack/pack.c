/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "../yaksa_test_config.h"
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

int main(int argc, char **argv)
{
    DTP_pool_s dtp;
    DTP_obj_s sobj, dobj;
    int rc;
    char typestr[MAX_DTP_BASESTRLEN + 1];
    int basecount = -1;
    int seed = -1;
    int iters = -1;
    int segments = -1;
    int pack_order = PACK_ORDER__UNSET;
    int overlap = -1;

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
        } else if (!strcmp(*argv, "-pack-order")) {
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
        } else {
            fprintf(stderr, "unknown argument %s\n", *argv);
            exit(1);
        }
    }
    if (typestr == NULL || basecount <= 0 || seed < 0 || iters < 0 || segments < 0 ||
        pack_order == PACK_ORDER__UNSET || overlap < 0) {
        fprintf(stderr, "Usage: ./pack {options}\n");
        fprintf(stderr, "   -datatype    base datatype to use, e.g., int\n");
        fprintf(stderr, "   -count       number of base datatypes in the signature\n");
        fprintf(stderr, "   -seed        random seed (changes the datatypes generated)\n");
        fprintf(stderr, "   -iters       number of iterations\n");
        fprintf(stderr, "   -segments    number of segments to chop the packing into\n");
        fprintf(stderr, "   -pack-order  packing order of segments (normal, reverse, random)\n");
        fprintf(stderr, "   -overlap     should packing overlap (none, regular, irregular)\n");
        exit(1);
    }

    yaksa_init();

    rc = DTP_pool_create(typestr, basecount, seed, &dtp);
    assert(rc == DTP_SUCCESS);

    uintptr_t *segment_starts = (uintptr_t *) malloc(segments * sizeof(uintptr_t));
    uintptr_t *segment_lengths = (uintptr_t *) malloc(segments * sizeof(uintptr_t));

    for (int i = 0; i < iters; i++) {
        char *desc;

        dprintf("==== iter %d ====\n", i);

        /* create the source object */
        rc = DTP_obj_create(dtp, &sobj, maxbufsize);
        assert(rc == DTP_SUCCESS);

        char *sbuf;
        sbuf = (char *) malloc(sobj.DTP_bufsize);
        assert(sbuf);

        if (verbose) {
            rc = DTP_obj_get_description(sobj, &desc);
            assert(rc == DTP_SUCCESS);
            dprintf("==> sbuf %p, sobj (count: %zu):\n%s\n", sbuf, sobj.DTP_type_count, desc);
        }

        rc = DTP_obj_buf_init(sobj, sbuf, 0, 1, basecount);
        assert(rc == DTP_SUCCESS);

        uintptr_t ssize;
        rc = yaksa_get_size(sobj.DTP_datatype, &ssize);
        assert(rc == YAKSA_SUCCESS);


        /* create the destination object */
        rc = DTP_obj_create(dtp, &dobj, maxbufsize);
        assert(rc == DTP_SUCCESS);

        char *dbuf;
        dbuf = (char *) malloc(dobj.DTP_bufsize);
        assert(dbuf);

        if (verbose) {
            rc = DTP_obj_get_description(dobj, &desc);
            assert(rc == DTP_SUCCESS);
            dprintf("==> dbuf %p, dobj (count: %zu):\n%s\n", dbuf, dobj.DTP_type_count, desc);
        }

        rc = DTP_obj_buf_init(dobj, dbuf, -1, -1, basecount);
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
        uintptr_t tbufsize = ssize * sobj.DTP_type_count;

#ifdef CUDA_SBUF
        char *sbuf_d, *orig_sbuf;
        cudaMalloc((void **) &sbuf_d, sobj.DTP_bufsize);
        assert(sbuf_d);
        cudaMemcpy(sbuf_d, sbuf, sobj.DTP_bufsize, cudaMemcpyHostToDevice);
        orig_sbuf = sbuf;
        sbuf = sbuf_d;
#endif

#ifdef CUDA_DBUF
        char *dbuf_d, *orig_dbuf;
        cudaMalloc((void **) &dbuf_d, dobj.DTP_bufsize);
        assert(dbuf_d);
        cudaMemcpy(dbuf_d, dbuf, dobj.DTP_bufsize, cudaMemcpyHostToDevice);
        orig_dbuf = dbuf;
        dbuf = dbuf_d;
#endif

        void *tbuf;
#ifdef CUDA_TBUF
        cudaMalloc((void **) &tbuf, tbufsize);
#else
        tbuf = malloc(tbufsize);
#endif

        for (int j = 0; j < segments; j++) {
            uintptr_t actual_pack_bytes;
            yaksa_request_t request;

            rc = yaksa_ipack(sbuf + sobj.DTP_buf_offset, sobj.DTP_type_count, sobj.DTP_datatype,
                             segment_starts[j], tbuf, segment_lengths[j], &actual_pack_bytes,
                             &request);
            assert(rc == YAKSA_SUCCESS);
            assert(actual_pack_bytes <= segment_lengths[j]);

            if (request != YAKSA_REQUEST__NULL) {
                rc = yaksa_request_wait(request);
                assert(rc == YAKSA_SUCCESS);
            }

            rc = yaksa_iunpack(tbuf, actual_pack_bytes, dbuf + dobj.DTP_buf_offset,
                               dobj.DTP_type_count, dobj.DTP_datatype, segment_starts[j], &request);
            assert(rc == YAKSA_SUCCESS);

            if (request != YAKSA_REQUEST__NULL) {
                rc = yaksa_request_wait(request);
                assert(rc == YAKSA_SUCCESS);
            }
        }

#ifdef CUDA_DBUF
        cudaMemcpy(orig_dbuf, dbuf, dobj.DTP_bufsize, cudaMemcpyDeviceToHost);
        dbuf = orig_dbuf;
#endif
        rc = DTP_obj_buf_check(dobj, dbuf, 0, 1, basecount);
        assert(rc == DTP_SUCCESS);


        /* free allocated buffers and objects */
#ifdef CUDA_SBUF
        cudaFree(sbuf_d);
        sbuf = orig_sbuf;
#endif
        free(sbuf);

#ifdef CUDA_DBUF
        cudaFree(dbuf_d);
#endif
        free(dbuf);

#ifdef CUDA_TBUF
        cudaFree(tbuf);
#else
        free(tbuf);
#endif

        DTP_obj_free(sobj);
        DTP_obj_free(dobj);
    }

    free(segment_lengths);
    free(segment_starts);
    DTP_pool_free(dtp);

    yaksa_finalize();

    return 0;
}
