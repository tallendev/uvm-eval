#!/bin/bash -xe

module load cuda

if [ ! -d "kokkos" ] || [ ! -d "kokkos-kernels" ]; then
    bash ./getKokkosKernels.sh
fi

CWD=`pwd`
KOKKOS_PATH="`pwd`/kokkos" #path to kokkos source
KOKKOSKERNELS_PATH="`pwd`/kokkos-kernels"          #path to kokkos-kernels top directory

# Compiler - must be passed to kokkos and kokkos-kernels configurations
CXX=${KOKKOS_PATH}/bin/nvcc_wrapper #Options: icpc #g++ #clang++
CXXFLAGS="-Wall -pedantic -Werror -O3 -g -Wshadow -Wsign-compare -Wignored-qualifiers -Wempty-body -Wclobbered -Wuninitialized"

# Configure Kokkos (Unit Tests OFF) - Makefile located in kokkos-build
cmake -Bkokkos-build -DCMAKE_CXX_COMPILER=${CXX} \
    -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_ENABLE_HWLOC=On \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
    -DCMAKE_INSTALL_PREFIX="${CWD}/kokkos-install" \
    -DKokkos_ARCH_VOLTA70:BOOL=ON \
    -DKokkos_ARCH_ZEN:BOOL=ON \
     -DKokkos_ENABLE_CUDA_UVM=ON \
    -DKokkos_ENABLE_EXAMPLES=ON \
    -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \
    -DKokkos_ENABLE_TESTS=OFF ${KOKKOS_PATH}

# Build and Install Kokkos - install lib at ${PWD}/kokkos-install
cmake --build kokkos-build -j 32 --target install


# Configure KokkosKernels (Unit Tests OFF) - Makefile located in kokkoskernels-build
cmake -Bkokkoskernels-build -DCMAKE_CXX_COMPILER=${CXX} -DKokkos_ROOT="${CWD}/kokkos-install" \
    -DKokkosKernels_INST_DOUBLE=ON -DKokkosKernels_INST_COMPLEX_DOUBLE=ON -DKokkosKernels_INST_ORDINAL_INT=ON \
    -DKokkosKernels_INST_ORDINAL_INT64_T=ON -DKokkosKernels_INST_OFFSET_INT=ON \
    -DKokkosKernels_INST_OFFSET_SIZE_T=ON -DKokkosKernels_INST_LAYOUTLEFT=ON -DKokkosKernels_ADD_DEFAULT_ETI=ON \
    -DCMAKE_INSTALL_PREFIX="${CWD}/kokkoskernels-install" -DKokkosKernels_ENABLE_TESTS=OFF -DKokkosKernels_ENABLE_EXAMPLES:BOOL=ON \
    -DKokkosKernels_ENABLE_TPL_CUBLAS=ON \
    -DKokkosKernels_CUBLAS_ROOT:PATH=/usr/local/cuda -DKokkosKernels_CUSPARSE_ROOT:PATH=/usr/local/cuda \
   ${KOKKOSKERNELS_PATH}
    
#-DKokkosKernels_REQUIRE_DEVICES=CUDA \
 #   -DKokkosKernels_REQUIRE_OPTIONS=cuda_relocatable_device_code \

cp KokkosSparse_wiki_gauss_seidel.cpp kokkos-kernels/example/wiki/sparse/
# Build and Install KokkosKernels - install lib at ${PWD}/kokkoskernels-install
cmake --build kokkoskernels-build -j 32 --target install VERBOSE=1

ln -fs kokkoskernels-build/example/wiki/sparse/KokkosKernels_wiki_gauss_seidel gs
