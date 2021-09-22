#!/bin/bash
source ../../scripts/common

FILE_MTX=./road_usa.mtx
FILE_ZIP=./road_usa.tar.gz

if [ ! -f "$FILE_MTX" ]; then
    if [ ! -f "$FILE_ZIP" ]; then
        wget $BASE_URL/road_usa.tar.gz
    fi
    tar zxf road_usa.tar.gz
    if [ $? -eq 1 ]; then
        echo "[CACTUS] Problem extracting road_usa.tar.gz"
        rm -f road_usa.tar.gz
        exit 1
    fi
    mv road_usa/road_usa.mtx .
    rm -rf road_usa/ road_usa.tar.gz    
fi

$GUN_HOME/1.1/build/bin/bfs market $FILE_MTX

# More information https://www.cise.ufl.edu/research/sparse/matrices/DIMACS10/road_usa.html
