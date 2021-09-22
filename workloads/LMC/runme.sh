#!/bin/bash
source ../../scripts/common
if [ ! -f in.colloid ]; then
    wget $BASE_URL/in.colloid
fi
$LMP_HOME/29Oct2020/src/lmp_mpi -sf gpu -in in.colloid

# More information https://docs.lammps.org/pair_colloid.html
