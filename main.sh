#!/bin/bash
clear
source ./venv/bin/activate
###########################################
## Not cross
###########################################
for l in {1W,1M}
do
  python3 ./generate_results.py --L=$l --models=ar
  python3 ./generate_results.py --L=$l --models=risk_metrics
  python3 ./generate_results.py --L=$l --models=har
  python3 ./generate_results.py --L=$l --models=har_dummy_markets
  python3 ./generate_results.py --L=$l --models=har_cdr
  python3 ./generate_results.py --L=$l --models=har_universal
done
###########################################
## Cross
###########################################
#for regression in {linear,ensemble}
#regression=ensemble
#do
#for l in {1D,1W,1M}
#do
#python3 ./generate_results.py --models=ar --L=1M
#python3 ./generate_results.py --models=har --L=1M
#python3 ./generate_results.py --models=har_cdr --L=1M
#python3 ./generate_results.py --models=har_universal --L=1M
#done
#done
for l in {1W,1M}
do
  python3 ./save_plots.py --L=$l
  #python3 ./save_plots.py --L=$l --test=True --cross=True
done
deactivate
