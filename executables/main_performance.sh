#!/bin/bash
clear
source ../venv/bin/activate
python3 ../generate_performance.py --n_performers=1
deactivate
#sqlite3 ../data_centre/databases/y.db < ../sql_scripts/performance.sql