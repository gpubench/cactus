#!/bin/bash
source ../../scripts/common

if [ ! -d game ]; then
  if [ ! -f game.zip ]; then
    wget $BASE_URL/game.zip
  fi
  unzip game.zip
  if [ $? -eq 1 ]; then
    echo "[CACTUS] Problem fetching game.zip"
    rm -f game.zip
    exit 1
  fi  
fi
if [ ! -d assets ]; then
  if [ ! -f assets.zip ]; then
    wget $BASE_URL/assets.zip
  fi
  unzip assets.zip
  if [ $? -eq 1 ]; then
    echo "[CACTUS] Problem fetching assets.zip"
    rm -f assets.zip
    exit 1
  fi   
fi

rm -f game.zip assets.zip

python3 main.py train

# More information https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch
