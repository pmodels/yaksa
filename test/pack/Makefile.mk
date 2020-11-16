##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

pack_testlists = $(top_srcdir)/test/pack/testlist.pack.gen \
	$(top_srcdir)/test/pack/testlist.pack.threads.gen \
	$(top_srcdir)/test/pack/testlist.acc.gen \
	$(top_srcdir)/test/pack/testlist.acc.threads.gen

EXTRA_DIST += $(top_srcdir)/test/pack/testlist.pack.gen \
	$(top_srcdir)/test/pack/testlist.pack.threads.gen \
	$(top_srcdir)/test/pack/testlist.acc.gen \
	$(top_srcdir)/test/pack/testlist.acc.threads.gen

EXTRA_PROGRAMS += \
	test/pack/pack \
	test/pack/acc

test_pack_pack_CPPFLAGS = $(test_cppflags)
test_pack_acc_CPPFLAGS = $(test_cppflags)

common_files = test/pack/pack-common.c \
	test/pack/pack-cuda.c   \
	test/pack/pack-ze.c

test_pack_pack_SOURCES = test/pack/pack.c ${common_files}
test_pack_acc_SOURCES = test/pack/acc.c ${common_files}

testlists += $(pack_testlists)
