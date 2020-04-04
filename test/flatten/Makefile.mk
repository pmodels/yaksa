##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

testlists += test/flatten/testlist.gen
EXTRA_DIST += test/flatten/testlist.gen

EXTRA_PROGRAMS += \
	test/flatten/flatten

test_flatten_flatten_CPPFLAGS = $(test_cppflags)
