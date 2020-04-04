##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

EXTRA_DIST = runtests.py

SUBDIRS = dtpools simple pack iov flatten

noinst_HEADERS =
include_HEADERS =
noinst_PROGRAMS =
nodist_noinst_SCRIPTS =
dist_noinst_SCRIPTS =

LDADD = $(top_builddir)/../libyaksa.la dtpools/libdtpools.la

testing:
	@$(top_srcdir)/runtests.py --summary=$(top_builddir)/summary.junit.xml \
                $(top_srcdir)/testlist
