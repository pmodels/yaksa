##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

AM_CPPFLAGS = -I$(top_builddir)/../src/frontend/include -I$(top_srcdir)/dtpools/src
EXTRA_DIST = testlist

LDADD = $(top_builddir)/dtpools/libdtpools.la $(top_builddir)/../libyaksa.la

noinst_PROGRAMS = \
	pack

pack_DEPENDENCIES = $(top_builddir)/dtpools/libdtpools.la $(top_builddir)/../libyaksa.la

testlists = $(top_srcdir)/pack/testlist.gen

if BUILD_CUDA_TESTS
include $(top_srcdir)/pack/Makefile.cuda.mk
endif BUILD_CUDA_TESTS

$(top_builddir)/dtpools/libdtpools.la:
	(cd $(top_builddir)/dtpools && $(MAKE))

testing:
	for tlist in ${testlists} ; do \
		$(top_srcdir)/runtests.py --testlist=$${tlist} \
			--summary=$(top_builddir)/pack/summary.junit.xml ; \
	done
