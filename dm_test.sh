#!/bin/bash
clear
source venv/bin/activate
#for L in {1D,1W,1M}
#do
  python3 ./generate_dm_test_table.py --L=1M --training_scheme=SAM &
#done
deactivate