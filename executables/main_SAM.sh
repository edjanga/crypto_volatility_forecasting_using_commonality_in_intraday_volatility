#!/bin/bash
clear
source ../venv/bin/activate
#######################################################
## Model performance SAM.
#######################################################
for l in {1W,1M,6M}
  do
    # for regression_type in {lightgbm,elastic,lasso,ridge,pcr,linear}
#      do
#       for model in {har,har_eq}
#        do
#           if [[ $model == "har_eq" ]]; then
#             python3 ../generate_results.py --L=$l --model=$model --training_scheme=SAM --regression=$regression_type \
#             --trading_session=0 --top_book=1
#             python3 ../generate_results.py --L=$l --model=$model --training_scheme=SAM --regression=$regression_type \
#             --trading_session=1
#           else
#             python3 ../generate_results.py --L=$l --model=$model --training_scheme=SAM --regression=$regression_type
#           fi
#        done
#      done
     #python3 ../generate_results.py --L=$l --model=ar --training_scheme=SAM --regression=linear
     python3 ../generate_results.py --L=$l --model=risk_metrics --training_scheme=SAM --regression=ewma
  done
#sqlite3 ../data_centre/databases/y.db < ../sql_scripts/export_table_SAM.sql
#zip ../results/y_SAM.zip ../results/y_SAM.csv
#zip ../results/qlike_SAM.zip ../results/qlike_SAM.csv
#rm ../results/*.csv