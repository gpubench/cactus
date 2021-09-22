#!/bin/bash
source ../../scripts/common

if [ ! -d MNIST ]; then
  if [ ! -f MNIST.tar.xz ]; then
    wget $BASE_URL/MNIST.tar.xz
  fi
  tar xf MNIST.tar.xz
  if [ $? -eq 1 ]; then
    echo "[CACTUS] Problem extracting MNIST.tar.xz"
    rm -f MNIST.tar.xz
    exit 1
  fi
  rm -f MNIST.tar.xz
fi

python3 main.py

# More information https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
