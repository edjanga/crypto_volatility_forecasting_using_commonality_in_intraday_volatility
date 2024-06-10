# apt apt install python3.10-venv
sudo apt-get install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
source ./venv/bin/activate
pip3 install -r requirements.txt
pip3 install wheel
python3 setup.py bdist_wheel
# Installation of ML framework for GPU (not lightgbm)
pip3 install --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.4.* dask-cudf-cu12==24.4.* cuml-cu12==24.4.* \
    cugraph-cu12==24.4.* cuspatial-cu12==24.4.* cuproj-cu12==24.4.* \
    cuxfilter-cu12==24.4.* cucim-cu12==24.4.* pylibraft-cu12==24.4.* \
    raft-dask-cu12==24.4.* cuvs-cu12==24.4.*
# git clone https://github.com/rapidsai/rapidsai-csp-utils.git
# python rapidsai-csp-utils/colab/pip-install.py
# Installation of ML framework for GPU (lightgbm)
# sudo apt-get install -y libboost-all-dev
# pip3 install lightgbm --install-option=--gpu
