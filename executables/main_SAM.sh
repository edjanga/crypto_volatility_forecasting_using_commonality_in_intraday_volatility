#!/bin/bash
clear
source ../venv/bin/activate
#######################################################
## Model performance SAM.
#######################################################
for l in {1W,1M,6M}
  do
    python3 ../generate_results.py --L=$l --model=risk_metrics --training_scheme=SAM --regression=ewma
    for regression_type in {lightgbm,lasso,elastic,ridge,pcr,linear}
      do
       for model in {har_eq,har}
        do
           if [[ $regression_type == "linear" ]] && [[ $model == "har_eq" ]]; then
             python3 ../generate_results.py --L=$l --model=$model --training_scheme=SAM --regression=$regression_type \
             --trading_session=0 --top_book=1
             python3 ../generate_results.py --L=$l --model=$model --training_scheme=SAM --regression=$regression_type \
             --trading_session=1
           elif [[ $regression_type != "linear" ]] && [[ $model == "har_eq" ]]; then
             python3 ../generate_results.py --L=$l --model=$model --training_scheme=SAM --regression=$regression_type \
             --trading_session=0 --top_book=1
             python3 ../generate_results.py --L=$l --model=$model --training_scheme=SAM --regression=$regression_type \
             --trading_session=1
           else
             python3 ../generate_results.py --L=$l --model=$model --training_scheme=SAM --regression=$regression_type
           fi
        done
      done
     python3 ../generate_results.py --L=$l --model=ar --training_scheme=SAM --regression=linear
  done