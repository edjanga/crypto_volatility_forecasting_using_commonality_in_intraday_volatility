apt apt install python3.10-venv
source ./venv/bin/activate
pip3 install -r requirements.txt
# Installation of ML framework for GPU (not lightgbm)
git clone https://github.com/rapidsai/rapidsai-csp-utils.git
python rapidsai-csp-utils/colab/pip-install.py
# Installation of ML framework for GPU (lightgbm)
apt-get install -y libboost-all-dev
pip3 install lightgbm --install-option=--gpu
