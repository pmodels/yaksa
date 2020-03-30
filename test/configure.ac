##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

AC_PREREQ(2.63)

AC_INIT([yaksa-tests],[0])

AC_CONFIG_AUX_DIR(m4)
AC_CONFIG_MACRO_DIR(m4)


dnl ----------------------------------------------------------------------------
dnl setup autotools
dnl ----------------------------------------------------------------------------
AM_INIT_AUTOMAKE([-Wall -Wno-portability-recursive -Werror foreign 1.12.3 subdir-objects])
AM_SILENT_RULES([yes])

PAC_PUSH_FLAG([CFLAGS])
AM_PROG_AS
PAC_POP_FLAG([CFLAGS])

AM_PROG_AR

LT_INIT

AC_PROG_INSTALL


dnl ----------------------------------------------------------------------------
dnl check the compiler and standard headers
dnl ----------------------------------------------------------------------------
dnl disable AC_PROG_CC's annoying default of adding -O2 to the CFLAGS
PAC_PUSH_FLAG([CFLAGS])
AC_PROG_CC
AC_PROG_CC_C99
PAC_POP_FLAG([CFLAGS])
AC_HEADER_STDC


dnl ----------------------------------------------------------------------------
dnl configure options
dnl ----------------------------------------------------------------------------
# strict
PAC_ARG_STRICT

# --enable-debug
AC_ARG_ENABLE([debug],AS_HELP_STRING([--enable-debug],[adds -g to CFLAGS]))
if test "${enable_debug}" = "yes" ; then
    PAC_APPEND_FLAG([-g],[CFLAGS])
fi

# --enable-fast
AC_ARG_ENABLE([fast],AS_HELP_STRING([--enable-fast=O<opt>],[adds -O<opt> to CFLAGS]),,[enable_fast=yes])
if test "${enable_fast}" = "yes" ; then
    PAC_APPEND_FLAG([-O2],[CFLAGS])
elif test "${enable_fast}" != "no" ; then
    PAC_APPEND_FLAG([-${enable_fast}],[CFLAGS])
fi


dnl ----------------------------------------------------------------------------
dnl check the environment and the availability of functions
dnl ----------------------------------------------------------------------------
PAC_C_GNU_ATTRIBUTE

# look for pthreads
AC_CHECK_HEADERS(pthread.h)
AC_CHECK_LIB([pthread],[pthread_key_create],have_pthreads=yes)
if test "$have_pthreads" = "yes" ; then
    PAC_PREPEND_FLAG([-lpthread],[LIBS])
else
    AC_ERROR([pthreads not found on the system])
fi

# look for CUDA
PAC_SET_HEADER_LIB_PATH([cuda])
PAC_CHECK_HEADER_LIB([cuda_runtime_api.h],[cudart],[cudaStreamSynchronize],[have_cuda=yes],[have_cuda=no])
if test "${have_cuda}" = "yes" ; then
    AC_DEFINE([HAVE_CUDA],[1],[Define is CUDA is available])
fi
AM_CONDITIONAL([BUILD_CUDA_TESTS], [test x${have_cuda} = xyes])


dnl ----------------------------------------------------------------------------
dnl Final setup
dnl ----------------------------------------------------------------------------


dnl ----------------------------------------------------------------------------
dnl config headers
dnl ----------------------------------------------------------------------------
AC_CONFIG_HEADERS([yaksa_test_config.h])
AH_TOP([
/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef YAKSA_TEST_CONFIG_H_INCLUDED
#define YAKSA_TEST_CONFIG_H_INCLUDED
])
AH_BOTTOM([
#endif /* YAKSA_TEST_CONFIG_H_INCLUDED */
])

AC_CONFIG_FILES([
	Makefile
	dtpools/Makefile
	flatten/Makefile
	iov/Makefile
	pack/Makefile
	simple/Makefile
])
AC_OUTPUT
