#!/bin/sh

# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and/or associated documentation files (the
# "Materials"), to deal in the Materials without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Materials, and to
# permit persons to whom the Materials are furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# unaltered in all copies or substantial portions of the Materials.
# Any additions, deletions, or changes to the original source files
# must be clearly indicated in accompanying documentation.
#
# If only executable code is distributed, then the accompanying
# documentation must state that "this software is based in part on the
# work of the Khronos Group."
#
# THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

##############################################################################
#
# check-libglvnd-install.sh
#
# This script checks for an existing libglvnd installation. A driver installer
# can use this to determine whether or not it needs to install its own copies
# of the libraries.
#
# The script will exit with one of these values:
# 0 -- The libglvnd libraries are installed and working.
# 1 -- No libglvnd libraries are installed.
# 2 -- There's a combination of libglvnd and non-libglvnd libraries installed.
# 3 -- Some internal error occurred.
#
##############################################################################


# Exit codes.
RESULT_INSTALLED=0
RESULT_NOT_INSTALLED=1
RESULT_PARTIAL=2
RESULT_ERROR=3

TEST_EXIT_CODE_SUCCESS=0
TEST_EXIT_CODE_NON_LIBGLVND=1
TEST_EXIT_CODE_VERSION_MISMATCH=2
TEST_EXIT_CODE_NO_LIBRARY=3
TEST_EXIT_CODE_INTERNAL_ERROR=4

BASEDIR=`readlink -f $0`
BASEDIR=`dirname $BASEDIR`
BINDIR=$BASEDIR
HELPER_PROGRAM=glvnd_check

__EGL_VENDOR_LIBRARY_FILENAMES=$BASEDIR/egl_dummy_vendor.json
__GLX_VENDOR_LIBRARY_NAME=installcheck
export __GLX_VENDOR_LIBRARY_NAME
export __EGL_VENDOR_LIBRARY_FILENAMES

if [ ! -e $__EGL_VENDOR_LIBRARY_FILENAMES ] ; then
    echo Missing test file: $__EGL_VENDOR_LIBRARY_FILENAMES
    exit $RESULT_ERROR
fi

MISSING_LIBRARIES=
INVALID_LIBRARIES=
LIBGLVND_LIBRARIES=
NON_LIBGLVND_LIBRARIES=

check_libglvnd_winsys()
{
    result=`LD_LIBRARY_PATH="${BINDIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$BINDIR/$HELPER_PROGRAM" $1 $2`
    code=$?

    case $code in
        $TEST_EXIT_CODE_SUCCESS) LIBGLVND_LIBRARIES="$2 $LIBGLVND_LIBRARIES" ;;
        $TEST_EXIT_CODE_NON_LIBGLVND) NON_LIBGLVND_LIBRARIES="$2 $NON_LIBGLVND_LIBRARIES" ;;
        $TEST_EXIT_CODE_NO_LIBRARY) MISSING_LIBRARIES="$2 $MISSING_LIBRARIES" ;;
        $TEST_EXIT_CODE_VERSION_MISMATCH) INVALID_LIBRARIES="$2 $INVALID_LIBRARIES" ;;
        *)
            echo Internal error:
            echo "$result"
            exit $RESULT_ERROR
    esac
    return $code
}

# Check the GLX libraries.
check_libglvnd_winsys glx libGL.so.1
check_libglvnd_winsys glx libGLX.so.0

# Check EGL and the entrypoint libraries. Note that checking the entrypoint
# libraries requires a libglvnd-based version of libEGL.so.1.
check_libglvnd_winsys egl libEGL.so.1
if [ "$?" -eq $TEST_EXIT_CODE_SUCCESS ] ; then
    check_libglvnd_winsys gl libOpenGL.so.0
    check_libglvnd_winsys gl libGLESv1_CM.so.1
    check_libglvnd_winsys gl libGLESv2.so.2
fi

echo Found libglvnd libraries: $LIBGLVND_LIBRARIES
echo Found non-libglvnd libraries: $NON_LIBGLVND_LIBRARIES
echo Missing libraries: $MISSING_LIBRARIES

if [ -n "$INVALID_LIBRARIES" ] ; then
    echo Found invalid or unsupported versions of libraries: $INVALID_LIBRARIES
    exit $RESULT_NOT_INSTALLED
fi

if [ -n "$LIBGLVND_LIBRARIES" ] ; then
    if [ -z "$NON_LIBGLVND_LIBRARIES" ] ; then
    # Some disros may split the libglvnd libraries into separate packages,
    # in which case it's possible that only some of them are installed.
    # As long as we found at least one libglvnd library, and we didn't find
    # any non-libglvnd versions, that's enough.
        echo libglvnd appears to be installed.
        exit $RESULT_INSTALLED
    else
    # There's a combination of libglvnd and non-libglvnd libraries
    # installed.
        exit $RESULT_PARTIAL
    fi
fi

# No libglvnd libraries are installed. Either the libraries are not present at
# all, or they're all non-libglvnd versions.
exit $RESULT_NOT_INSTALLED
