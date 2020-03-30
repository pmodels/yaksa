##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

include $(top_srcdir)/src/backend/cuda/include/Makefile.mk
include $(top_srcdir)/src/backend/cuda/hooks/Makefile.mk
include $(top_srcdir)/src/backend/cuda/md/Makefile.mk
include $(top_srcdir)/src/backend/cuda/pup/Makefile.mk

GENCODE_FLAGS = -gencode arch=compute_$(CUDA_SM),code=sm_$(CUDA_SM)
.cu.lo:
	@if $(AM_V_P) ; then \
		$(top_srcdir)/cudalt.sh --verbose $@ $(NVCC) $(AM_CPPFLAGS) $(GENCODE_FLAGS) -c $< ; \
	else \
		echo "  NVCC     $@" ; \
		$(top_srcdir)/cudalt.sh $@ $(NVCC) $(AM_CPPFLAGS) $(GENCODE_FLAGS) -c $< ; \
	fi
