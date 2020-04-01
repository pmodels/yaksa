##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

noinst_PROGRAMS += \
	pack_cuda_sbuf \
	pack_cuda_dbuf \
	pack_cuda_tbuf \
	pack_cuda_sbuf_dbuf \
	pack_cuda_sbuf_tbuf \
	pack_cuda_dbuf_tbuf \
	pack_cuda_sbuf_dbuf_tbuf

pack_cuda_sbuf_SOURCES = pack.c
pack_cuda_sbuf_CPPFLAGS = -DCUDA_SBUF $(AM_CPPFLAGS)

pack_cuda_dbuf_SOURCES = pack.c
pack_cuda_dbuf_CPPFLAGS = -DCUDA_DBUF $(AM_CPPFLAGS)

pack_cuda_tbuf_SOURCES = pack.c
pack_cuda_tbuf_CPPFLAGS = -DCUDA_TBUF $(AM_CPPFLAGS)

pack_cuda_sbuf_dbuf_SOURCES = pack.c
pack_cuda_sbuf_dbuf_CPPFLAGS = -DCUDA_SBUF -DCUDA_DBUF $(AM_CPPFLAGS)

pack_cuda_sbuf_tbuf_SOURCES = pack.c
pack_cuda_sbuf_tbuf_CPPFLAGS = -DCUDA_SBUF -DCUDA_TBUF $(AM_CPPFLAGS)

pack_cuda_dbuf_tbuf_SOURCES = pack.c
pack_cuda_dbuf_tbuf_CPPFLAGS = -DCUDA_DBUF -DCUDA_TBUF $(AM_CPPFLAGS)

pack_cuda_sbuf_dbuf_tbuf_SOURCES = pack.c
pack_cuda_sbuf_dbuf_tbuf_CPPFLAGS = -DCUDA_SBUF -DCUDA_DBUF -DCUDA_TBUF $(AM_CPPFLAGS)

testlists += $(top_srcdir)/pack/testlist.cuda.gen
