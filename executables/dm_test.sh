#!/bin/bash
clear
source ../venv/bin/activate
python3 ../generate_dm_test_table.py --test=1
deactivate

