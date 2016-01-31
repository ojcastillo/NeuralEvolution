#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.68])
AC_INIT([NeuralEvolution], [1.0], [jcm.orlando@gmail.com])
AC_CONFIG_SRCDIR([src/NeuralEvolution.c])
AC_CONFIG_HEADERS([config.h])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_PROG_RANLIB

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC

# Checks for mpi libraries.

# If --with-mpi=no is used, use a standard C compiler instead.
AC_ARG_WITH(mpi, 
[
	AS_HELP_STRING([--with-mpi],
	[compilar con soporte para MPI (paralelizacion). Si no se encuentra,
	MPI no es usado. Default: no])
]
,,[with_mpi=no])

#Verificacion
AX_PROG_CC_MPI([test x"$with_mpi" != xno],[use_mpi=yes],[
	use_mpi=no
	if test x"$with_mpi" = xyes; then
		AC_MSG_FAILURE([Compilacion con MPI solicitada, pero no se pudo encontrar.])
	else
		AC_MSG_WARN([No se solicito el soporte para MPI.])
	fi
])

#Check for math library
AC_CHECK_LIB(m, sqrt) 

# Checks for header files.
AC_CHECK_HEADERS([inttypes.h limits.h stdlib.h string.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_TYPE_PID_T
AC_TYPE_UINT64_T

# Checks for library functions.
AC_FUNC_FORK
AC_FUNC_MALLOC
AC_FUNC_STRTOD

AC_CONFIG_FILES(
[
           Makefile
           src/Makefile
           lib/Makefile
])
AC_OUTPUT
