#!/bin/bash
clear
source ../venv/bin/activate
#######################################################
## Commonality for different time horizons.
#######################################################
python ../generate_commonality.py --commonality_type=absorption_ratio --title_figure=0
python ../generate_commonality.py --commonality_type=adjusted_r2 --title_figure=0
deactivate
