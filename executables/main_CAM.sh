#!/bin/bash
clear
source ../venv/bin/activate
#######################################################
## Computations for CAM.
#######################################################
model=har_eq
for l in {1W,1M,6M}
  do
    for regression_type in {lightgbm,elastic,lasso,ridge,pcr}
      do
         #for model in {har,har_eq,ar}
            #do
               if [ $model == "har_eq" ]; then
                  python3 ../generate_results.py --L=$l --model=$model --training_scheme=CAM \
                  --regression=$regression_type --trading_session=1
                  python3 ../generate_results.py --L=$l --model=$model --training_scheme=CAM \
                  --regression=$regression_type --trading_session=0 --top_book=1
               else
                 python3 ../generate_results.py --L=$l --model=$model --training_scheme=CAM \
                 --regression=$regression_type
               fi
            #done
      done
  done
#sqlite3 ../data_centre/databases/y.db < ../sql_scripts/export_table_CAM.sql
#zip ../results/y_CAM.zip ../results/y_CAM.csv
#zip ../results/qlike_CAM.zip ../results/qlike_CAM.csv
#rm ../results/*.csv
