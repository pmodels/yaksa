##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

EXTRA_PROGRAMS += \
	test/simple/simple_test \
	test/simple/threaded_test

test_simple_simple_test_CPPFLAGS = $(test_cppflags)
test_simple_simple_test_LDADD = $(test_ldadd)

test_simple_threaded_test_CPPFLAGS = $(test_cppflags)
test_simple_threaded_test_LDADD = $(test_ldadd)

test_simple: 
	@$(top_srcdir)/test/runtests.py --summary=$(builddir)/test/simple/summary.junit.xml \
                $(top_srcdir)/test/simple/testlist.gen
