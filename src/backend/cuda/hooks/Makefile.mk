##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

AM_CPPFLAGS += -I$(top_srcdir)/src/backend/cuda/hooks

libyaksa_la_SOURCES += \
	src/backend/cuda/hooks/yaksuri_cuda_init_hooks.c \
	src/backend/cuda/hooks/yaksuri_cuda_type_hooks.c
