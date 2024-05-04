#!/bin/bash
clear
source ../venv/bin/activate
python3 ../generate_backtesting.py --liquidity=.9 --min_expiration=7D --max_expiration=180D --performance=returns \
--to_latex=1 --save=1