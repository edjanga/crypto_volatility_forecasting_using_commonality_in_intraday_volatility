#!/bin/bash
clear
source ../venv/bin/activate
#######################################################
## Commonality for different time horizons.
#######################################################
# python ../generate_spillover.py --best_performing=1 --spillover_network=0 --spillover_tsi=1 --node_degrees_balance=0 \
# --title_figure=0
#python ../generate_spillover.py --best_performing=0 --spillover_network=1 --spillover_tsi=0 --node_degrees_balance=1 \
#--title_figure=0
python ../generate_spillover.py --best_performing=0 --spillover_network=0 --spillover_tsi=1 --node_degrees_balance=0 \
--title_figure=0
#python ../generate_spillover.py --best_performing=0 --spillover_network=0 --spillover_tsi=0 --node_degrees_balance=1 \
#--title_figure=0
deactivate