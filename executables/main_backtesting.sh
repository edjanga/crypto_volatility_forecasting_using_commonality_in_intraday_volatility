#!/bin/bash
clear
source ../venv/bin/activate
python3 ../generate_backtesting.py --liquidity=.75 --min_expiration=7D --max_expiration=180D --performance=returns \
--save=0