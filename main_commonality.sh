#!/bin/bash
clear
source ./venv/bin/activate
#######################################################
## Commonality for different time horizons.
#######################################################
generate=True
save=True
if [ "$generate" = "True" ]
then
  for L in {1D,1W,1M}
    do
      python ./generate_commonality.py --L=$L --transformation=log --generate=$generate --save=False
    done
fi
generate=False
python ./generate_commonality.py --transformation=log --generate=$generate --save=$save


deactivate
