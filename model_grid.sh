#!/bin/bash
clear
source venv/bin/activate
python3 ./generate_model_grid.py --training_scheme=ClustAM
deactivate