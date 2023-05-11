#!/bin/bash
clear
if [[ "$USER" = 'emmanueldjanga' ]];
  then
    dir_to_tardis=/Volumes/TOSHIBA\ EXT
else
  dir_to_tardis=/data/cholgpu01/not-backed-up/datasets/graf/data
fi
source ./venv/bin/activate
python3 ./data_centre/tardis_initial_data.py --destination "$dir_to_tardis/TARDISData"
#python3 ./data_centre/minutely_tardis.py --location "$dir_to_tardis/TARDISData"
deactivate