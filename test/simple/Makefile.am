##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

AM_CPPFLAGS = -I$(top_builddir)/../src/frontend/include -I$(top_srcdir)/dtpools/src
EXTRA_DIST = testlist

LDADD = $(top_builddir)/dtpools/libdtpools.la $(top_builddir)/../libyaksa.la

noinst_PROGRAMS = \
	simple_test \
	threaded_test

$(top_builddir)/dtpools/libdtpools.la:
	(cd $(top_builddir)/dtpools && $(MAKE))

testing:
	@$(top_srcdir)/runtests.py --testlist=$(top_srcdir)/simple/testlist \
		--summary=$(top_builddir)/simple/summary.junit.xml
