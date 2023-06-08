#!/bin/bash
clear
source ./venv/bin/activate
python3 ./model/lab.py
python3 ./plots/maker.py
deactivate 
