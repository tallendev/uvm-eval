#!/bin/bash
# Requires cmake version > 3.12
# Paths to source

module load cuda

KOKKOS_RELEASE=3.4.01
wget https://github.com/kokkos/kokkos/archive/refs/tags/${KOKKOS_RELEASE}.zip
unzip ${KOKKOS_RELEASE}.zip
rm -r ${KOKKOS_RELEASE}.zip
mv kokkos-${KOKKOS_RELEASE} kokkos

wget https://github.com/kokkos/kokkos-kernels/archive/refs/tags/${KOKKOS_RELEASE}.zip
unzip ${KOKKOS_RELEASE}.zip
rm -r ${KOKKOS_RELEASE}.zip
mv kokkos-kernels-${KOKKOS_RELEASE} kokkos-kernels



