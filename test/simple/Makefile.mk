##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

testlists += test/simple/testlist.gen
EXTRA_DIST += test/simple/testlist.gen

EXTRA_PROGRAMS += \
	test/simple/simple_test \
	test/simple/threaded_test

test_simple_simple_test_CPPFLAGS = $(test_cppflags)
test_simple_threaded_test_CPPFLAGS = $(test_cppflags)
