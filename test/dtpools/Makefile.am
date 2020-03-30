##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

AM_CPPFLAGS = -I$(top_srcdir)/src -I$(top_builddir)/../src/frontend/include

LDADD = $(top_builddir)/../../libyaksa.la

noinst_LTLIBRARIES = libdtpools.la
libdtpools_la_SOURCES = \
	src/dtpools.c \
	src/dtpools_attr.c \
	src/dtpools_desc.c \
	src/dtpools_init_verify.c \
	src/dtpools_misc.c

noinst_HEADERS = \
	src/dtpools.h \
	src/dtpools_internal.h
