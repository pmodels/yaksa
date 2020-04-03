##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

EXTRA_PROGRAMS += \
	test/iov/iov

test_iov_iov_CPPFLAGS = $(test_cppflags)
test_iov_iov_LDADD = $(test_ldadd)

test_iov: 
	@$(top_srcdir)/test/runtests.py --summary=$(builddir)/test/iov/summary.junit.xml \
                $(top_srcdir)/test/iov/testlist.gen
