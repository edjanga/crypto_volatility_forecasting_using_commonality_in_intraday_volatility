#!/bin/bash
clear
source ../venv/bin/activate
#######################################################
## Commonality for different time horizons.
#######################################################
python ../generate_commonality.py --commonality_type=absorption_ratio
python ../generate_commonality.py --commonality_type=adjusted_r2
deactivate
