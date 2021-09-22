#!/bin/bash
source ../../scripts/common
if [ ! -f npt.tpr ]; then
    wget $BASE_URL/npt.tpr
fi
$GMX_HOME/2021/bin/gmx mdrun -notunepme -pme gpu -nb gpu -v -deffnm npt

# More information http://www.mdtutorials.com/gmx/complex/index.html
