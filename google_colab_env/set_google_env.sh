apt install python3.10-venv
source ./venv/bin/activate
pip3 install -r requirements.txt
# Build LightGBM for GPU
/bin/bash build_lightgbm_gpu.sh