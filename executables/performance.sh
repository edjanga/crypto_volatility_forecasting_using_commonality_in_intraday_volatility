#!/bin/bash
clear
source ../venv/bin/activate
sqlite3 -header -csv "../data_centre/databases/qlike.db" < ../sql_scripts/performance.sql > ../results/performance.csv
python3 ../generate_performance.py --n_performers=1 --bottom_performers=0
deactivate