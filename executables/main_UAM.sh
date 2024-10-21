#!/bin/bash
clear
source ../venv/bin/activate
#######################################################
## Computations for UAM.
#######################################################
#l=1M
#for regression_type in {lstm,lightgbm}
#  do
#      for model in {har_eq,ar,har}
#        do
#          if [ $model == "har_eq" ]; then
#            python3 ../generate_results.py --L=$l --model=$model --training_scheme=UAM --regression=$regression_type \
#            --trading_session=0 --top_book=1
#            python3 ../generate_results.py --L=$l --model=$model --training_scheme=UAM --regression=$regression_type \
#            --trading_session=1
#          else
#            python3 ../generate_results.py --L=$l --model=$model --training_scheme=UAM --regression=$regression_type
#          fi
#        done
#  done

#for l in {6M,1M,1W}
#  do
#    # python3 ../generate_results.py --L=$l --training_scheme=UAM --regression=var
#    for regression_type in {lstm,lightgbm}
#    do
#      for model in {ar,har,har_eq}
#        do
#          if [ $model == "har_eq" ]; then
#            python3 ../generate_results.py --L=$l --model=$model --training_scheme=UAM --regression=$regression_type \
#            --trading_session=0 --top_book=1
#            python3 ../generate_results.py --L=$l --model=$model --training_scheme=UAM --regression=$regression_type \
#            --trading_session=1
#          else
#            python3 ../generate_results.py --L=$l --model=$model --training_scheme=UAM --regression=$regression_type
#          fi
#        done
#    done
#  done
sqlite3 ../data_centre/databases/y.db < ../sql_scripts/export_table_UAM_1W.sql &
sqlite3 ../data_centre/databases/y.db < ../sql_scripts/export_table_UAM_1M.sql &
sqlite3 ../data_centre/databases/y.db < ../sql_scripts/export_table_UAM_6M.sql &
wait
zip ../results/y_UAM_1W.zip ../results/y_UAM_1W.csv &
zip ../results/y_UAM_1M.zip ../results/y_UAM_1M.csv &
zip ../results/y_UAM_6M.zip ../results/y_UAM_6M.csv &
rm ../results/*.csv