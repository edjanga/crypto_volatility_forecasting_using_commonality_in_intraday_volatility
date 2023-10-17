#!/bin/bash
clear
sqlite3 -header -csv "./data_centre/databases/r2.db" < ./sql_scripts/performance.sql > performance.csv
source venv/bin/activate
python3 ./generate_performance.py
deactivate