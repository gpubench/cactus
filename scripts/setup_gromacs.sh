#!/bin/bash

cd $GMX_HOME
if [ -f $GMX_HOME/.installed ]; then
    echo "[CACTUS] Skipping Gromacs."
    exit 0
fi

echo "[CACTUS] Fetching and building Gromacs."

if [ ! -f gromacs-2021.tar.gz ]; then
    wget $BASE_URL/gromacs-2021.tar.gz
    # Alternative download link 
    # https://manual.gromacs.org/documentation/ 
fi
tar zxf gromacs-2021.tar.gz
if [ $? -eq 1 ]; then
    echo "[CACTUS] There is a problem with the tar file. Maybe it is broken. Please run again to fetch the file."
    rm -f gromacs-2021.tar.gz
    exit 1
fi

mv gromacs-2021 2021
cd 2021
if [ ! -d build ]; then
    mkdir build
fi
cd build
cmake .. -DGMX_GPU=CUDA \
         -DCMAKE_INSTALL_PREFIX=.. \
         -DGMX_BUILD_OWN_FFTW=ON \
         -DGMX_CUDA_TARGET_SM=$ARCH
make -j8
make install -j8

GMX_BIN=../bin/gmx
if [ -f "$GMX_BIN" ]; then
    echo "[CACTUS] Gromacs installed successfully."
    touch $GMX_HOME/.installed
    exit 0
else
    echo "[CACTUS] There was a problem building Gromacs."
    exit 1
fi
