#!/bin/bash
source ../../scripts/common

if [ ! -d celeba ]; then
  if [ ! -f img_align_celeba.zip ]; then
    wget $BASE_URL/img_align_celeba.zip
  fi
  unzip -d celeba img_align_celeba.zip
  if [ $? -eq 1 ]; then
    echo "[CACTUS] Problem extracting img_align_celeba.zip"
    rm -f img_align_celeba.zip
    exit 1
  fi
  rm -f img_align_celeba.zip
fi

python3 main.py

# More information https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

