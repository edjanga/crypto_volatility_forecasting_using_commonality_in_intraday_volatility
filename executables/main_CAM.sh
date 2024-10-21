#!/bin/bash
clear
source ../venv/bin/activate
#######################################################
## Computations for CAM.
#######################################################
#model=har_eq
#for l in {1W,1M,6M}
#  do
#    for regression_type in {lightgbm,elastic,lasso,ridge,pcr}
#      do
#         #for model in {har,har_eq,ar}
#            #do
#               if [ $model == "har_eq" ]; then
#                  python3 ../generate_results.py --L=$l --model=$model --training_scheme=CAM \
#                  --regression=$regression_type --trading_session=1
#                  python3 ../generate_results.py --L=$l --model=$model --training_scheme=CAM \
#                  --regression=$regression_type --trading_session=0 --top_book=1
#               else
#                 python3 ../generate_results.py --L=$l --model=$model --training_scheme=CAM \
#                 --regression=$regression_type
#               fi
#            #done
#      done
#  done
sqlite3 ../data_centre/databases/y.db < ../sql_scripts/export_table_CAM_1W.sql &
sqlite3 ../data_centre/databases/y.db < ../sql_scripts/export_table_CAM_1M.sql &
sqlite3 ../data_centre/databases/y.db < ../sql_scripts/export_table_CAM_6M.sql &
wait
zip ../results/y_CAM_1W.zip ../results/y_CAM_1W.csv &
zip ../results/y_CAM_1M.zip ../results/y_CAM_1M.csv &
zip ../results/y_CAM_6M.zip ../results/y_CAM_6M.csv &
rm ../results/*.csv
