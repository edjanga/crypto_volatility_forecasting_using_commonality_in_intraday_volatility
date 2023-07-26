#!/bin/bash
clear
sqlite3 -header -csv "./data_centre/databases/r2.db" < performance.sql > performance.csv
source venv/bin/activate
python3 ./performance.py
deactivate