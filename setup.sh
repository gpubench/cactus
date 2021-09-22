#!/bin/bash

source ./scripts/common

## Running workloads on real GPU needs the Nvidia driver to be installed. 
## We prefer to install the Nvidia driver and toolkit from the the .run file rather than the Ubuntu apt packages.
## We use the 460.27 driver that comes with CUDA-11.2.
## For installing the driver, the Nouveau kernel driver must be disabled. Otherwise the Nvidia installer will fail.
## If you have a working `nvidia-smi` and `nvcc`, it should be fine to run the workloads.
## However, you should note that requirements provided by Nvidia. 
## For example, if you have an Ampere device, you must use CUDA > 11.
## Another thing to consider is that 460.27 driver fails to compile on Ubuntu 20.04.3 that comes with kernel 5.11
## We have tested the installation with Ubuntu 20.04.1 that has kernel 5.4.
## In this case, you may want to manually 
##     install linux-headers-5.4.0-42 
##     linux-headers-5.4.0-42-generic 
##     linux-image-5.4.0-42-generic
## And boot the system with 5.4 kernel.

OS_VERSION=`awk -F= '$1=="VERSION" {gsub(/"/, "", $2); print $2}' /etc/os-release`
KERNEL_VERSION=`uname -r`
VGA_DEVICES=`lspci | grep -i --color 'vga\|3d\|2d'`
NVIDIA_VERSION=`modinfo /usr/lib/modules/$(uname -r)/kernel/drivers/video/nvidia.ko | grep ^version | awk '{print $2}'`
NVCC_VERSION=`nvcc --version | sed -n 's/^.*release \([0-9\.]\+\).*$/\1/p'`

echo "[CACTUS] Welcome! "
echo "[CACTUS] We have tested Cactus on Ubuntu 20.04.1 with kernel 5.4.0-42, Nvidia RTX 3080 (SM_86), CUDA 11.2 and driver 460.27"
echo "[CACTUS] A summary of your system information is printed below"
echo "[CACTUS] OS version: $OS_VERSION"
echo "[CACTUS] Kernel version: $KERNEL_VERSION"
echo "[CACTUS] VGA devices: $VGA_DEVICES"
echo "[CACTUS] Nvidia driver version: $NVIDIA_VERSION"
echo "[CACTUS] nvcc compiler version: $NVCC_VERSION"

if [ $INSTALL_CUDA = "YES" ]; then
    if [ ! -f .nouveau_blacklisted ]; then
        echo "[CACTUS] I Will disable Nouveau and reboot and then install 460.27 driver and CUDA-11.2 toolkit."
        sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/nvidia-installer-disable-nouveau.conf"
        sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/nvidia-installer-disable-nouveau.conf"
        sudo update-initramfs -u -k all
        touch .nouveau_blacklisted
        echo "[CACTUS] Please reboot the system and run setup again".
        exit 0
    else
        if [ ! -f cuda_11.2.0_460.27.04_linux.run ]; then
            wget $BASE_URL/cuda_11.2.0_460.27.04_linux.run
            chmod +x cuda_11.2.0_460.27.04_linux.run
        fi  
        echo "[CACTUS] Installing CUDA-11.2"
        sudo ./cuda_11.2.0_460.27.04_linux.run --driver --toolkit --silent
        if [ $? -eq 1 ]; then
             echo "[CACTUS] Problem installing CUDA-11.2 on kernel `uname -r`."
             exit 1
        fi
        echo "export PATH=/usr/local/cuda/bin:$PATH" >> $HOME/.bashrc
        echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> $HOME/.bashrc
        source $HOME/.bashrc
    fi
fi


## Install some apt packages
if [ ! -f .installed_apt_packages ]; then
    echo "[CACTUS] Installing some basic apt packages with sudo access."
    sudo apt install -y python3-pip cmake g++ gcc build-essential zlib1g-dev git
    touch .installed_apt_packages
fi

./scripts/setup_gromacs.sh
if [ $? -eq 1 ]; then
     echo "[CACTUS] setup_gromacs.sh exited with errors."
     exit 1
fi
./scripts/setup_lammps.sh
if [ $? -eq 1 ]; then
     echo "[CACTUS] setup_lammps.sh exited with errors."
     exit 1
fi
./scripts/setup_gunrock.sh
if [ $? -eq 1 ]; then
     echo "[CACTUS] setup_gunrock.sh exited with errors."
     exit 1
fi
./scripts/setup_pytorch.sh
if [ $? -eq 1 ]; then
     echo "[CACTUS] setup_pytorch.sh exited with errors."
     exit 1
fi

echo "[CACTUS] Programs have been installed. You can now execute \"runme.sh\" files in ./workloads/"
