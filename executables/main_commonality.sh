#!/bin/bash
clear
source ../venv/bin/activate
#######################################################
## Commonality for different time horizons.
#######################################################
for L in {1W,1M,6M}
  do
    python ../generate_commonality.py --L=$L --transformation=log --generate=1 --save=0
  done
python ../generate_commonality.py --transformation=log --generate=0 --save=1
deactivate
