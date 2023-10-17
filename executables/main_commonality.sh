#!/bin/bash
clear
source ./venv/bin/activate
#######################################################
## Commonality for different time horizons.
#######################################################
python ./generate/generate_commonality.py --L=$L --transformation=log --generate=1 --save=0
python ./generate/generate_commonality.py --transformation=log --generate=0 --save=1
deactivate
