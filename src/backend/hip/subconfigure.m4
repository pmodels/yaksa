##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##


##########################################################################
##### capture user arguments
##########################################################################

# --with-hip-sm
AC_ARG_WITH([hip-sm],
            [
  --with-hip-sm=<options> (https://reviews.llvm.org/differential/changeset/?ref=2126544)
          Comma-separated list of below options:
                all - build compatibility for all GPUs supported by the HIP version (can increase compilation time)

                # Kepler architecture
                AMDGCN - build compatibility for all AMD GPUs
                gfx600 - GFX600-DAG
                gfx601 - GFX601-DAG
                gfx700 - GFX700-DAG
                gfx701 - GFX701-DAG
                gfx702 - GFX702-DAG
                gfx703 - GFX703-DAG
                gfx704 - GFX704-DAG
                gfx801 - GFX801-DAG
                gfx802 - GFX802-DAG
                gfx803 - GFX803-DAG
                gfx810 - GFX810-DAG
                gfx900 - GFX900-DAG
                gfx902 - GFX902-DAG
                gfx904 - GFX904-DAG
                gfx906 - GFX906-DAG
                gfx908 - GFX908-DAG
                gfx909 - GFX909-DAG
                gfx1010 - GFX1010-DAG
                gfx1011 - GFX1011-DAG
                gfx1012 - GFX1012-DAG
                gfx1030 - GFX1030-DAG
                gfx1031 - GFX1031-DAG
                # Other
                <numeric> - specific SM numeric to use
            ],,
            [with_hip_sm=all])


# --with-hip
PAC_SET_HEADER_LIB_PATH([hip])
if test "$with_hip" != "no" ; then
    PAC_PUSH_FLAG(CPPFLAGS)
    PAC_APPEND_FLAG([-D__HIP_PLATFORM_AMD__], [CPPFLAGS])
    PAC_CHECK_HEADER_LIB([hip/hip_runtime_api.h],[amdhip64],[hipStreamSynchronize],[have_hip=yes],[have_hip=no])
    PAC_POP_FLAG(CPPFLAGS)
    if test "${have_hip}" = "yes" ; then
        AC_MSG_CHECKING([whether hipcc works])
        cat>conftest.c<<EOF
        #include <hip/hip_runtime_api.h>
        void foo(int deviceId) {
        hipError_t ret;
        ret = hipGetDevice(&deviceId);
        }
EOF
        PAC_PUSH_FLAG(CPPFLAGS)
        PAC_APPEND_FLAG([-D__HIP_PLATFORM_AMD__], [CPPFLAGS])
        ${with_hip}/bin/hipcc -c conftest.c 2> /dev/null
        PAC_POP_FLAG(CPPFLAGS)
        if test "$?" = "0" ; then
            AC_DEFINE([HAVE_HIP],[1],[Define is HIP is available])
            PAC_APPEND_FLAG([-D__HIP_PLATFORM_AMD__], [CPPFLAGS])
            AS_IF([test -n "${with_hip}"],[HIPCC=${with_hip}/bin/hipcc],[HIPCC=hipcc])
            AC_SUBST(HIPCC)
            # hipcc compiled applications need libstdc++ to be able to link
            # with a C compiler
            PAC_PUSH_FLAG([LIBS])
            PAC_APPEND_FLAG([-lstdc++],[LIBS])
            AC_LINK_IFELSE(
                [AC_LANG_PROGRAM([int x = 5;],[x++;])],
                [libstdcpp_works=yes],
                [libstdcpp_works=no])
            PAC_POP_FLAG([LIBS])
            if test "${libstdcpp_works}" = "yes" ; then
                PAC_APPEND_FLAG([-lstdc++],[LIBS])
                AC_MSG_RESULT([yes])
            else
                have_hip=no
                AC_MSG_RESULT([no])
                AC_MSG_ERROR([hipcc compiled applications need libstdc++ to be able to link with a C compiler])
            fi
        else
            have_hip=no
            AC_MSG_RESULT([no])
        fi
        rm -f conftest.*
    fi
fi
AM_CONDITIONAL([BUILD_HIP_BACKEND], [test x${have_hip} = xyes])

# --with-hip-p2p
AC_ARG_ENABLE([hip-p2p],AS_HELP_STRING([--enable-hip-p2p={yes|no|cliques}],[controls HIP P2P capability]),,
              [enable_hip_p2p=yes])
if test "${enable_hip_p2p}" = "yes" ; then
    AC_DEFINE([HIP_P2P],[HIP_P2P_ENABLED],[Define if HIP P2P is enabled])
elif test "${enable_hip_p2p}" = "cliques" ; then
    AC_DEFINE([HIP_P2P],[HIP_P2P_CLIQUES],[Define if HIP P2P is enabled in clique mode])
else
    AC_DEFINE([HIP_P2P],[HIP_P2P_DISABLED],[Define if HIP P2P is disabled])
fi


##########################################################################
##### analyze the user arguments and setup internal infrastructure
##########################################################################

if test "${have_hip}" = "yes" ; then
    for maj_version in 4 3 2 1; do
        version=$((maj_version * 1000))
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([
                              #include <hip/hip_runtime.h>
                              int x[[HIP_VERSION - $version]];
                          ],)],[hip_version=${maj_version}],[])
        if test ! -z ${hip_version} ; then break ; fi
    done
    PAC_PUSH_FLAG([IFS])
    IFS=","
    HIP_SM=
    for sm in ${with_hip_sm} ; do
        case "$sm" in
            all|AMDGCN)
                HIP_SM="gfx600 gfx601 gfx700 gfx701 gfx702 gfx703 gfx704 gfx801 gfx802 gfx803 gfx810 gfx900 gfx902 gfx904 gfx906 gfx908 gfx909 gfx1010 gfx1011 gfx1012 gfx1030 gfx1031"
                ;;
            gfx6)
                PAC_APPEND_FLAG([gfx600],[HIP_SM])
                PAC_APPEND_FLAG([gfx601],[HIP_SM])
                ;;
            gfx7)
                PAC_APPEND_FLAG([gfx701],[HIP_SM])
                PAC_APPEND_FLAG([gfx702],[HIP_SM])
                PAC_APPEND_FLAG([gfx703],[HIP_SM])
                PAC_APPEND_FLAG([gfx704],[HIP_SM])
                ;;
            gfx8)
                PAC_APPEND_FLAG([gfx801],[HIP_SM])
                PAC_APPEND_FLAG([gfx802],[HIP_SM])
                PAC_APPEND_FLAG([gfx803],[HIP_SM])
                ;;
            gfx9)
                PAC_APPEND_FLAG([gfx900],[HIP_SM])
                PAC_APPEND_FLAG([gfx902],[HIP_SM])
                PAC_APPEND_FLAG([gfx904],[HIP_SM])
                PAC_APPEND_FLAG([gfx906],[HIP_SM])
                PAC_APPEND_FLAG([gfx908],[HIP_SM])
                PAC_APPEND_FLAG([gfx909],[HIP_SM])
                ;;
            gfx10.1)
                PAC_APPEND_FLAG([gfx1010],[HIP_SM])
                PAC_APPEND_FLAG([gfx1011],[HIP_SM])
                PAC_APPEND_FLAG([gfx1012],[HIP_SM])
                ;;
            gfx10.3)
                PAC_APPEND_FLAG([gfx1030],[HIP_SM])
                PAC_APPEND_FLAG([gfx1031],[HIP_SM])
                ;;
            none)
                ;;
            *)
                PAC_APPEND_FLAG([$sm], [HIP_SM])
        esac
    done
    PAC_POP_FLAG([IFS])

    for sm in ${HIP_SM} ; do
        if test -z "${HIP_GENCODE}" ; then
            HIP_GENCODE="--offload-arch=${sm}"
        else
            HIP_GENCODE="${HIP_GENCODE} --offload-arch=${sm}"
        fi
    done
    AC_SUBST(HIP_GENCODE)
    if test -z "${HIP_GENCODE}" ; then
        AC_MSG_ERROR([--with-hip-sm not specified; either specify it or disable hip support])
    fi

    supported_backends="${supported_backends},hip"
    backend_info="${backend_info}
HIP backend specific options:
      HIP GENCODE: ${with_hip_sm} (${HIP_GENCODE})"
fi
