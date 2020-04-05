##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

pack_testlists += test/pack/testlist.cuda.gen
EXTRA_DIST += test/pack/testlist.cuda.gen

EXTRA_PROGRAMS += \
	test/pack/pack_cuda_sbuf \
	test/pack/pack_cuda_dbuf \
	test/pack/pack_cuda_tbuf \
	test/pack/pack_cuda_sbuf_dbuf \
	test/pack/pack_cuda_sbuf_tbuf \
	test/pack/pack_cuda_dbuf_tbuf \
	test/pack/pack_cuda_sbuf_dbuf_tbuf

test_pack_pack_cuda_sbuf_SOURCES = test/pack/pack.c
test_pack_pack_cuda_sbuf_CPPFLAGS = -DCUDA_SBUF $(test_cppflags)

test_pack_pack_cuda_dbuf_SOURCES = test/pack/pack.c
test_pack_pack_cuda_dbuf_CPPFLAGS = -DCUDA_DBUF $(test_cppflags)

test_pack_pack_cuda_tbuf_SOURCES = test/pack/pack.c
test_pack_pack_cuda_tbuf_CPPFLAGS = -DCUDA_TBUF $(test_cppflags)

test_pack_pack_cuda_sbuf_dbuf_SOURCES = test/pack/pack.c
test_pack_pack_cuda_sbuf_dbuf_CPPFLAGS = -DCUDA_SBUF -DCUDA_DBUF $(test_cppflags)

test_pack_pack_cuda_sbuf_tbuf_SOURCES = test/pack/pack.c
test_pack_pack_cuda_sbuf_tbuf_CPPFLAGS = -DCUDA_SBUF -DCUDA_TBUF $(test_cppflags)

test_pack_pack_cuda_dbuf_tbuf_SOURCES = test/pack/pack.c
test_pack_pack_cuda_dbuf_tbuf_CPPFLAGS = -DCUDA_DBUF -DCUDA_TBUF $(test_cppflags)

test_pack_pack_cuda_sbuf_dbuf_tbuf_SOURCES = test/pack/pack.c
test_pack_pack_cuda_sbuf_dbuf_tbuf_CPPFLAGS = -DCUDA_SBUF -DCUDA_DBUF -DCUDA_TBUF $(test_cppflags)
