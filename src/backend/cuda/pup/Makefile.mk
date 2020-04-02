##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

AM_CPPFLAGS += -I$(top_srcdir)/src/backend/cuda/pup

libyaksa_la_SOURCES += \
	src/backend/cuda/pup/yaksuri_cuda_request.c \
	src/backend/cuda/pup/yaksuri_cuda_get_memory_type.c \
	src/backend/cuda/pup/yaksuri_cudai_pup_char.cu \
	src/backend/cuda/pup/yaksuri_cudai_pup_double.cu \
	src/backend/cuda/pup/yaksuri_cudai_pup_float.cu \
	src/backend/cuda/pup/yaksuri_cudai_pup_int.cu \
	src/backend/cuda/pup/yaksuri_cudai_pup_int16_t.cu \
	src/backend/cuda/pup/yaksuri_cudai_pup_int32_t.cu \
	src/backend/cuda/pup/yaksuri_cudai_pup_int64_t.cu \
	src/backend/cuda/pup/yaksuri_cudai_pup_int8_t.cu \
	src/backend/cuda/pup/yaksuri_cudai_pup_long.cu \
	src/backend/cuda/pup/yaksuri_cudai_pup_long_long.cu \
	src/backend/cuda/pup/yaksuri_cudai_pup_short.cu \
	src/backend/cuda/pup/yaksuri_cudai_pup_wchar_t.cu \
	src/backend/cuda/pup/yaksuri_cudai_pup.c

noinst_HEADERS += \
	src/backend/seq/pup/yaksuri_seqi_pup.h
