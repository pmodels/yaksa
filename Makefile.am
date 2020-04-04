##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

ACLOCAL_AMFLAGS = -I m4

EXTRA_DIST = autogen.sh cudalt.sh

.PHONY: doxygen

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = maint/yaksa.pc

noinst_HEADERS =
include_HEADERS =
noinst_PROGRAMS =
nodist_noinst_SCRIPTS =
dist_noinst_SCRIPTS = autogen.sh

lib_LTLIBRARIES = libyaksa.la
libyaksa_la_SOURCES =
AM_CPPFLAGS =
libyaksa_la_LDFLAGS = -version-info @libyaksa_so_version@

include $(top_srcdir)/src/Makefile.mk

testlists=
include $(top_srcdir)/test/Makefile.mk

doxygen:
	mkdir -p doc
	doxygen Doxyfile

testing:
	@$(top_srcdir)/test/runtests.py --summary=$(top_builddir)/test/summary.junit.xml \
                $(testlists)
