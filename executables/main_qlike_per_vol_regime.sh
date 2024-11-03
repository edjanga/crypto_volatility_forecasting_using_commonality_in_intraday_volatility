#!/bin/bash
clear
source ../venv/bin/activate
sqlite3 ../data_centre/databases/y.db < ../sql_scripts/qlike_per_vol_regime.sql
deactivate
