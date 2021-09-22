#!/bin/bash

cd $PYT_HOME
if [ -f .installed_pytorch ] && [ -f .installed_text ] && [ -f .installed_vision ]; then
    echo "[CACTUS] Skipping Pytorch."
    exit 0
fi

echo "[CACTUS] Fetching and building Pytorch."

if [ $INSTALL_CUDNN = "YES" ]; then
    echo "[CACTUS] Installing CuDNN-8.1.0."
    if [ ! -f cudnn-11.2-linux-x64-v8.1.0.77.tgz ]; then
        wget $BASE_URL/cudnn-11.2-linux-x64-v8.1.0.77.tgz
    # Alternatively you can download from Nvidia website
    fi
    tar xf cudnn-11.2-linux-x64-v8.1.0.77.tgz
    if [ $? -eq 1 ]; then
        echo "[CACTUS] There is a problem with the tar file. Maybe it is broken. Please run again to fetch the file."
        rm -f cudnn-11.2-linux-x64-v8.1.0.77.tgz
        exit 1
    else
        sudo mv ./cuda/include/* /usr/local/cuda/include
        sudo mv ./cuda/lib64/* /usr/local/cuda/lib64
        rm -rf ./cuda
        echo "[CACTUS] CuDNN-8.1.0 has been installed successfully."
    fi
fi


## Install some pip3 packages
if [ ! -f $PYT_HOME/.installed_pip3_packages ]; then
    echo "[CACTUS] Installing some pip3 packages."
    pip3 install --user numpy pyyaml typing keras scipy future matplotlib typing_extensions ipython opencv-python pygame spacy
    touch $PYT_HOME/.installed_pip3_packages
fi

if [ -f $PYT_HOME/.installed_pytorch ]; then
    echo "[CACTUS] Skipping Pytorch."
else
    echo "[CACTUS] Installing PyTorch 1.7.1"
    if [ ! -f pytorch.20210702.tar.bz2 ]; then
        wget $BASE_URL/pytorch.20210702.tar.bz2
    fi
    tar xf pytorch.20210702.tar.bz2
    if [ $? -eq 1 ]; then
        echo "[CACTUS] There is a problem with the tar file. Maybe it is broken. Please run again to fetch the file."
        rm -f pytorch.20210702.tar.bz2
        exit 1
    fi
    mv pytorch pt.1.7.1
    cd pt.1.7.1
    python3 setup.py install --user
    if [ $? -eq 1 ]; then
        echo "[CACTUS] Problem installing Pytorch."
        exit 1
    else
        echo "[CACTUS] Pytorch installed successfully."
        touch $PYT_HOME/.installed_pytorch
    fi
fi

if [ -f $PYT_HOME/.installed_text ]; then
    echo "[CACTUS] Skipping Text."
else
    echo "[CACTUS] Installing Text 0.8.1"
    if [ ! -f torchtext.20210702.tar.bz2 ]; then
        wget $BASE_URL/torchtext.20210702.tar.bz2
    fi
    tar xf torchtext.20210702.tar.bz2
    if [ $? -eq 1 ]; then
        echo "[CACTUS] There is a problem with the tar file. Maybe it is broken. Please run again to fetch the file."
        rm -f torchtext.20210702.tar.bz2
        exit 1
    fi
    mv torchtext tt.0.8.1  
    cd tt.0.8.1 
    python3 setup.py install --user
    if [ $? -eq 1 ]; then
        echo "[CACTUS] Problem installing Text."
        exit 1
    else
        echo "[CACTUS] Text installed successfully."
        touch $PYT_HOME/.installed_text
    fi   
fi



if [ ! -f $PYT_HOME/.installed_vision ]; then
    echo "[CACTUS] Skipping Vision."
else
    echo "[CACTUS] Installing Vision 0.9"
    if [ ! -f vision.20211302.tar.bz2 ]; then
        wget $BASE_URL/vision.20211302.tar.bz2
    fi
    tar xf vision.20211302.tar.bz2
    if [ $? -eq 1 ]; then
        echo "[CACTUS] There is a problem with the tar file. Maybe it is broken. Please run again to fetch the file."
        rm -f vision.20211302.tar.bz2
    exit 1
    fi
    mv vision v.0.9  
    cd v.0.9 
    python3 setup.py install --user
    if [ $? -eq 1 ]; then
        echo "[CACTUS] Problem installing Vision."
        exit 1
    else
        echo "[CACTUS] Vision installed successfully."
        touch $PYT_HOME/.installed_vision
    fi     
fi




