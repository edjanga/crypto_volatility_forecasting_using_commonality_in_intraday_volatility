#!/bin/bash
clear
source ../venv/bin/activate
#######################################################
## Computations for UAM.
#######################################################
for l in {1W,1M,6M}
  do
    # python3 ../generate_results.py --L=$l --training_scheme=UAM --regression=var
    for regression_type in {lightgbm,lstm}
      do
        for model in {ar,har,har_eq}
          do
            if [ $model == "har_eq" ]; then
              python3 ../generate_results.py --L=$l --model=$model --training_scheme=UAM --regression=$regression_type \
              --trading_session=0 --top_book=1
              python3 ../generate_results.py --L=$l --model=$model --training_scheme=UAM --regression=$regression_type \
              --trading_session=1
            else
              python3 ../generate_results.py --L=$l --model=$model --training_scheme=UAM --regression=$regression_type
            fi
          done
      done
  done
sqlite3 ../data_centre/databases/y.db < ../sql_scripts/export_table_UAM.sql
zip ../results/y_UAM.zip ../results/y_UAM.csv
zip ../results/qlike_UAM.zip ../results/qlike_UAM.csv
rm ../results/*.csv