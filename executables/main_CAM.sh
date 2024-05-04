#!/bin/bash
clear
source ../venv/bin/activate
#######################################################
## Computations for CAM.
#######################################################
for l in {1W,1M,6M}
  do
    for regression_type in {lightgbm,elastic,lasso,ridge,pcr}
      do
         for model in {har,ar}
            do
               if [ $model == "har_eq" ]; then
                  python3 ../generate_results.py --L=$l --model=$model --training_scheme=CAM \
                  --regression=$regression_type --trading_session=1
                  python3 ../generate_results.py --L=$l --model=$model --training_scheme=CAM \
                  --regression=$regression_type --trading_session=0 --top_book=1
               else
                 python3 ../generate_results.py --L=$l --model=$model --training_scheme=CAM \
                 --regression=$regression_type
               fi
            done
      done
  done

