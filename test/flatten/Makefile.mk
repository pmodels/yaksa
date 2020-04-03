##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

EXTRA_PROGRAMS += \
	test/flatten/flatten

test_flatten_flatten_CPPFLAGS = $(test_cppflags)
test_flatten_flatten_LDADD = $(test_ldadd)

test_flatten: 
	@$(top_srcdir)/test/runtests.py --summary=$(builddir)/test/flatten/summary.junit.xml \
                $(top_srcdir)/test/flatten/testlist.gen
