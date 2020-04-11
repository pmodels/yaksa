##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

# --with-cuda-sm
AC_ARG_WITH([cuda-sm],AS_HELP_STRING([--with-cuda-sm=<numeric>],[builds CUDA support for that architecture]),,
            [with_cuda_sm=none])
if test "${with_cuda_sm}" != "none" ; then
    CUDA_SM="${with_cuda_sm}"
    AC_SUBST(CUDA_SM)
fi

# --with=cuda
PAC_SET_HEADER_LIB_PATH([cuda])
PAC_CHECK_HEADER_LIB([cuda_runtime_api.h],[cudart],[cudaStreamSynchronize],[have_cuda=yes],[have_cuda=no])
if test "${have_cuda}" = "yes" ; then
    AC_DEFINE([HAVE_CUDA],[1],[Define is CUDA is available])
    AS_IF([test -n "${with_cuda}"],[NVCC=${with_cuda}/bin/nvcc],[NVCC=nvcc])
    AC_SUBST(NVCC)

    if test -z "${CUDA_SM}" ; then
        AC_MSG_ERROR([--with-cuda-sm not specified; either specify it or disable cuda support])
    fi

    supported_backend="${supported_backends},cuda"
    backend_info="${backend_info}
CUDA backend specific options:
      CUDA SM: ${with_cuda_sm}"
fi
AM_CONDITIONAL([BUILD_CUDA_BACKEND], [test x${have_cuda} = xyes])
AM_CONDITIONAL([BUILD_CUDA_TESTS], [test x${have_cuda} = xyes])

# --with-cuda-p2p
AC_ARG_ENABLE([cuda-p2p],AS_HELP_STRING([--enable-cuda-p2p={yes|no|cliques}],[controls CUDA P2P capability]),,
              [enable_cuda_p2p=yes])
if test "${enable_cuda_p2p}" = "yes" ; then
    AC_DEFINE([CUDA_P2P],[CUDA_P2P_ENABLED],[Define if CUDA P2P is enabled])
elif test "${enable_cuda_p2p}" = "cliques" ; then
    AC_DEFINE([CUDA_P2P],[CUDA_P2P_CLIQUES],[Define if CUDA P2P is enabled in clique mode])
else
    AC_DEFINE([CUDA_P2P],[CUDA_P2P_DISABLED],[Define if CUDA P2P is disabled])
fi
