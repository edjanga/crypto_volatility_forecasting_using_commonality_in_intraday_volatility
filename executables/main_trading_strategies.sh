#!/bin/bash
clear
source ../venv/bin/activate
# Generate performance to get performance ranking
# /bin/bash ./performance.sh
# Backtest
python3 ../generate_backtesting.py --save=0 --min_expiration=0D --max_expiration=1D
deactivate
