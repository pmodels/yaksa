##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

EXTRA_PROGRAMS += \
	test/pack/pack

test_pack_pack_CPPFLAGS = $(test_cppflags)
test_pack_pack_LDADD = $(test_ldadd)

pack_testlists = $(top_srcdir)/test/pack/testlist.gen

if BUILD_CUDA_TESTS
include $(top_srcdir)/test/pack/Makefile.cuda.mk
endif BUILD_CUDA_TESTS

test_pack: 
	@$(top_srcdir)/test/runtests.py --summary=$(builddir)/test/pack/summary.junit.xml \
                $(pack_testlists)
