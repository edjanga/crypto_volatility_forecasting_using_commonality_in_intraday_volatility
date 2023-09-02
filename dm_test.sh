#!/bin/bash
clear
source venv/bin/activate
#for training_scheme in {SAM,ClustAM,CAM}
#  done
  for L in {1W,1M}
    do
      python3 ./generate_dm_test_table.py --L=$L --training_scheme=SAM
    done
#  done
deactivate