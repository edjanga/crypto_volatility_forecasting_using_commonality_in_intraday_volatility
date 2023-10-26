#!/bin/bash
clear
source ../venv/bin/activate
#######################################################
## Model performance ClustAM.
#######################################################
l=6M
regression_type=pcr
for model in {har_eq,har,ar}
  do
    python3 ../generate_results.py --L=$l --model=$model --training_scheme=ClustAM --regression=$regression_type \
    --transformation=log
  done
#for l in {1M,1W}
#  do
#    for regression_type in {lightgbm,elastic,lasso,pcr}
#      do
#        for model in {har_eq,har,ar}
#          do
#            python3 ../generate_results.py --L=$l --model=$model --training_scheme=ClustAM \
#            --regression=$regression_type --transformation=log &
#          done
#          wait
#      done
#  done
