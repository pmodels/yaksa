##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

EXTRA_DIST += runtests.py

LDADD = test/dtpools/libdtpools.la libyaksa.la
test_cppflags = -I$(build_dir)/src/frontend/include -I$(srcdir)/test/dtpools/src

include $(top_srcdir)/test/dtpools/Makefile.mk

EXTRA_PROGRAMS =
include $(top_srcdir)/test/simple/Makefile.mk
include $(top_srcdir)/test/pack/Makefile.mk
include $(top_srcdir)/test/iov/Makefile.mk
include $(top_srcdir)/test/flatten/Makefile.mk

CLEANFILES = $(EXTRA_PROGRAMS)
