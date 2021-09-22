#!/bin/bash
source ../../scripts/common

if [ ! -d images ]; then
  if [ ! -f images.zip ]; then
    wget $BASE_URL/images.zip
  fi
  unzip images.zip
  if [ $? -eq 1 ]; then
    echo "[CACTUS] Problem extracting images.zip"
    rm -f images.zip
    exit 1
  fi
  rm -f images.zip
fi

CKPT=$HOME/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth
if [ ! -f $CKPT ]; then
    wget $BASE_URL/vgg19-dcbb9e9d.pth -O $CKPT
    if [ $? -eq 1 ]; then
        echo "[CACTUS] Problem fetching vgg19-dcbb9e9d.pth"
        rm -f $CKPT
       exit 1
    fi   
fi

python3 main.py

# More information https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
