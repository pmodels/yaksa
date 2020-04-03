##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

EXTRA_DIST += test/runtests.py

SUBDIRS = test/dtpools

test_cppflags = -I$(top_builddir)/src/frontend/include -I$(top_srcdir)/test/dtpools/src
test_ldadd = $(top_builddir)/libyaksa.la $(top_builddir)/test/dtpools/libdtpools.la
EXTRA_PROGRAMS =

if BUILD_CUDA_TESTS
# some testlist are optional, e.g. test/pack/testlist.cuda.gen
# runtests.py uses environment variables to optionally run these tests
HAS_CUDA = 1
export HAS_CUDA
endif BUILD_CUDA_TESTS

include test/simple/Makefile.mk
include test/pack/Makefile.mk
include test/iov/Makefile.mk
include test/flatten/Makefile.mk

testing:
	@$(top_srcdir)/test/runtests.py --summary=$(builddir)/test/summary.junit.xml \
                $(srcdir)/test/testlist
