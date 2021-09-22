#!/bin/bash
source ../../scripts/common
if [ ! -f data.rhodo ]; then
    wget $BASE_URL/data.rhodo
fi
if [ ! -f in.rhodo ]; then
    wget $BASE_URL/in.rhodo
fi
$LMP_HOME/29Oct2020/src/lmp_mpi -sf gpu -in in.rhodo

# More information https://docs.lammps.org/Speed_bench.html
