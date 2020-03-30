##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

AM_CPPFLAGS += -I$(top_srcdir)/src/frontend/pools

libyaksa_la_SOURCES += \
	src/frontend/pools/yaksi_type_pool.c \
	src/frontend/pools/yaksi_request_pool.c
