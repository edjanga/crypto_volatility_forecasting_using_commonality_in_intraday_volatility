#!/bin/bash
clear
source ./venv/bin/activate
###########################################
## Not cross
###########################################
for l in {1D,1W,1M}
do
  python3 ./main.py --L=$l --test=True
done
###########################################
## Cross
###########################################
#for regression in {linear,ensemble}
#do
#  for l in {1D,1W,1M}
#  do
#    python3 ./main.py --cross=True --var_explained=.9 --models=har --L=$l --regression_type=$regression
#    python3 ./main.py --cross=True --var_explained=.9 --models=har --L=$l --regression_type=$regression
#    python3 ./main.py --cross=True --var_explained=.9 --models=har_cdr --L=$l --regression_type=$regression
#    python3 ./main.py --cross=True --var_explained=.9 --models=har_universal --L=$l --regression_type=$regression
#  done
#done
deactivate
