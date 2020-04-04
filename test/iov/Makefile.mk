##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

testlists += test/iov/testlist.gen
EXTRA_DIST += test/iov/testlist.gen

EXTRA_PROGRAMS += \
	test/iov/iov

test_iov_iov_CPPFLAGS = $(test_cppflags)

test-iov:
	@$(top_srcdir)/test/runtests.py --summary=$(top_builddir)/test/iov/summary.junit.xml \
                test/iov/testlist.gen
