#!/bin/bash

cd $LMP_HOME
if [ -f $LMP_HOME/.installed ]; then
    echo "[CACTUS] Skipping Lammps."
    exit 0
fi

echo "[CACTUS] Fetching and building Lammps."

if [ ! -f .installed_apt_packages ]; then
    echo "[CACTUS] Installing some apt packages with sudo access."
    sudo apt install -y libopenmpi-dev libjpeg-dev libpng-dev libfftw3-dev
    touch .installed_apt_packages
fi
if [ ! -f stable_29Oct2020.tar.gz ]; then
    wget $BASE_URL/stable_29Oct2020.tar.gz
    # Alternative download link 
    # https://download.lammps.org/tars/lammps-stable.tar.gz
fi

tar zxf stable_29Oct2020.tar.gz
if [ $? -eq 1 ]; then
    echo "[CACTUS] There is a problem with the tar file. Maybe it is broken. Please run again to fetch the file"
    rm -f stable_29Oct2020.tar.gz
    exit 1
fi

mv lammps-stable_29Oct2020 29Oct2020     
cd 29Oct2020/lib/gpu
sed -Ei "/(^|[^# ])CUDA_ARCH/ s/sm_.*/sm_$ARCH/g" Makefile.linux
make -f Makefile.linux
cd ../../src
make yes-gpu
make yes-molecule
make yes-kspace
make yes-rigid
make yes-colloid
make mpi -j8

LMP_BIN=./lmp_mpi
if [ -f "$LMP_BIN" ]; then
    echo "[CACTUS] Gromacs installed successfully"
    touch $LMP_HOME/.installed
    exit 0
else
    echo "[CACTUS] There was a problem building Lammps"
    exit 1
fi
