##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

AM_CPPFLAGS += -I$(top_srcdir)/src/backend/src

libyaksa_la_SOURCES += \
	src/backend/src/yaksur_hooks.c \
	src/backend/src/yaksur_pup.c

noinst_HEADERS += \
	src/backend/src/yaksur.h
