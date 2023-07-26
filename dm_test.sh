#!/bin/bash
clear
source venv/bin/activate
python3 ./generate_dm_test_table.py --L=1W --training_scheme=SAM
deactivate