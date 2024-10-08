#!/bin/bash
clear
source ../venv/bin/activate
python3 ../generate_backtesting.py --min_expiration=7D --max_expiration=180D --best_performing=1 --signal_strength=1.5 \
--title_figure=0
python3 ../generate_backtesting.py --min_expiration=7D --max_expiration=180D --best_performing=0 --signal_strength=1.5 \
--title_figure=0
deactivate