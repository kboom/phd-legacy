AC_PREREQ([2.64])

AC_INIT([bspline_2d], [1.0])

AC_CONFIG_SRCDIR([src/main.cc])
AC_CONFIG_MACRO_DIR([m4])

AC_PROG_CXX

AM_INIT_AUTOMAKE([foreign])

AC_CONFIG_HEADERS([include/config.h])

CUDA_INSTALL_PATH="/usr/local/cuda"
AC_ARG_WITH([cuda],
    [AS_HELP_STRING([--with-cuda=PATH],
                   [directory where CUDA is installed @<:@default=auto@:>@])],
    [CUDA_INSTALL_PATH=$withval],
    [with_cuda=auto])
NVCC="$CUDA_INSTALL_PATH/bin/nvcc"
AC_SUBST(NVCC)

CUDA_SDK_INSTALL_PATH="/usr/local/cuda-7.5"
AC_ARG_WITH([cuda-sdk],
   [AS_HELP_STRING(
        [--with-cuda-sdk=PATH],
        [directory where CUDA SDK is installed @<:@default=auto@:>@])],
   [CUDA_SDK_INSTALL_PATH=$withval],
   [with_cuda_sdk=auto])
cuda_CPPFLAGS="-I${CUDA_SDK_INSTALL_PATH}/include"
cuda_LIBS="-lcublas -lcuda"
AC_SUBST(cuda_CPPFLAGS)
AC_SUBST(cuda_LIBS)

COMPILER_DIR=/usr/bin
AC_ARG_WITH([compiler-dir],
   [AS_HELP_STRING(
        [--with-compiler-dir=PATH],
        [directory where GNU compilers are located @<:@default=auto@:>@])],
   [COMPILER_DIR=$withval],
   [with_compiler_dir=auto])
CXX=${COMPILER_DIR}/g++
AC_SUBST(CXX)

MAGMA_DIR="/usr/local/src/magma_1.0.0"
AC_ARG_WITH([magma-dir],
   [AS_HELP_STRING(
        [--with-magma-dir=PATH],
        [directory where MAGMA is installed @<:@default=auto@:>@])],
   [MAGMA_DIR=$withval],
   [with_magma_dir=auto])
magma_CPPFLAGS="-I${MAGMA_DIR}/include"
magma_LIBS="-L${MAGMA_DIR}/lib -lmagma"
AC_SUBST(magma_CPPFLAGS)
AC_SUBST(magma_LIBS)

lapack_LIBS="-llapack"
AC_SUBST(lapack_LIBS)

MUMPS_DIR="/usr/local/src/MUMPS"
AC_ARG_WITH([mumps-dir],
   [AS_HELP_STRING(
        [--with-mumps-dir=PATH],
        [directory where MUMPS is installed @<:@default=auto@:>@])],
   [MUMPS_DIR=$withval],
   [with_mumps_dir=auto])
mumps_CPPFLAGS="-I${MUMPS_DIR}/include"
mumps_LIBS="-L${MUMPS_DIR}/lib -L${MUMPS_DIR}/libseq -ldmumps -lmpiseq -lmumps_common -lpord -lblas -lpthread -lgfortran"
AC_SUBST(mumps_CPPFLAGS)
AC_SUBST(mumps_LIBS)


USER_LIB="~/usr"
AC_ARG_WITH([user-lib],
   [AS_HELP_STRING(
        [--with-user-lib=PATH],
        [include local include directory @<:@default=auto@:>@])],
   [USER_LIB=$withval],
   [with_user_lib=auto])
user_CPPFLAGS="-I${USER_LIB}/include"
user_LIBS="-I${USER_LIB}/lib"
AC_SUBST(user_CPPFLAGS)
AC_SUBST(user_LIBS)


AC_ARG_WITH([single-precision],
   [AS_HELP_STRING([--with-single-precision],
                   [Use single precision instead of double])])
AS_IF([test "x$with_single_precision" != "xyes"],
      [AC_DEFINE([FEM_PRECISION], [double], [Calculations precision])],
      [AC_DEFINE([FEM_PRECISION], [float], [Calculations precision])])

ARCH="-gencode arch=compute_20,\\\"code=sm_20,compute_20\\\""
NVCCFLAGS="${ARCH} --compiler-bindir ${COMPILER_DIR} -Xcompiler \"-Wall\" --ptxas-options=\"-v\""
AC_SUBST(NVCCFLAGS)

AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT
