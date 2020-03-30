##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

AM_CPPFLAGS += -I$(top_srcdir)/src/frontend/include -I$(top_builddir)/src/frontend/include

include_HEADERS += \
	src/frontend/include/yaksa.h

noinst_HEADERS += \
	src/frontend/include/yaksi.h
