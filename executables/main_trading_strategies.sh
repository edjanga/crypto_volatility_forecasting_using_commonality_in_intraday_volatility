#!/bin/bash
clear
source ../venv/bin/activate
# Generate performance to get performance ranking
# /bin/bash ./performance.sh
# Backtest
python3 ../generate_backtesting.py
deactivate