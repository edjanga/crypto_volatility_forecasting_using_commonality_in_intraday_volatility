#!/bin/bash
clear
source ../venv/bin/activate
#######################################################
## Model performance SAM.
#######################################################
 for l in {6M,1W,1M}
   do
     for regression_type in {lightgbm,elastic,lasso,pcr}
       do
         for model in {har_eq,har}
           do
             python3 ../generate_results.py --L=$l --model=$model --training_scheme=SAM \
             --regression=$regression_type --transformation=log &
             python3 ../generate_results.py --L=$l --model=$model --training_scheme=SAM --regression=linear \
             --transformation=log &
          done
          wait
      done
    python3 ../generate_results.py --L=$l --model=ar --training_scheme=SAM --regression=linear --transformation=log
  done

