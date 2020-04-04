##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

AM_CPPFLAGS = -I$(top_builddir)/../src/frontend/include -I$(top_srcdir)/dtpools/src
EXTRA_DIST = testlist

LDADD = $(top_builddir)/dtpools/libdtpools.la $(top_builddir)/../libyaksa.la

noinst_PROGRAMS = \
	iov

$(top_builddir)/dtpools/libdtpools.la:
	(cd $(top_builddir)/dtpools && $(MAKE))

testing:
	@$(top_srcdir)/runtests.py --summary=$(top_builddir)/iov/summary.junit.xml \
                $(top_srcdir)/iov/testlist.gen
