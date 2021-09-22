#!/bin/bash

cd $GUN_HOME
if [ -f $GUN_HOME/.installed ]; then
    echo "[CACTUS] Skipping Gunrock."
    exit 0
fi

echo "[CACTUS] Fetching and building Gunrock."

if [ ! -f .installed_apt_packages ]; then
    echo "[CACTUS] Installing some apt packages with sudo access."
    sudo apt install -y libboost-all-dev libmetis-dev rapidjson-dev
    touch .installed_apt_packages
fi
if [ ! -f gunrock.20210216.tar.xz ]; then
    wget $BASE_URL/gunrock.20210216.tar.xz
fi

tar xf gunrock.20210216.tar.xz
if [ $? -eq 1 ]; then
    echo "[CACTUS] There is a problem with the tar file. Maybe it is broken. Please run again to fetch the file."
    rm -f gunrock.20210216.tar.xz
    exit 1
fi

mv gunrock 1.1       
cd 1.1/build
cmake -DGUNROCK_BUILD_APPLICATIONS=OFF -DGUNROCK_APP_BFS=ON -DGUNROCK_GENCODE_SM$ARCH=ON ..
make -j8

BFS_BIN=../build/bin/bfs
if [ -f "$BFS_BIN" ]; then
    echo "[CACTUS] Gromacs installed successfully."
    touch $GUN_HOME/.installed
    exit 0
else
    echo "[CACTUS] There was a problem building Gunrock."
    exit 1
fi
