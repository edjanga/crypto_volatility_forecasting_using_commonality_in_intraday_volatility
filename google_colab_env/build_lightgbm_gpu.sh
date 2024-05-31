apt-get install -y --no-install-recommends build-essential libboost-dev libboost-system-dev libboost-filesystem-dev \
    libboost-chrono-dev libboost-thread-dev libboost-regex-dev libboost-program-options-dev libboost-python-dev \
    libboost-serialization-dev cmake libopenmpi-dev libomp-dev
apt-get install -y ocl-icd-libopencl1 ocl-icd-opencl-dev clinfo
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM && mkdir build && cd build && cmake -DUSE_GPU=1 .. && make -j$(nproc)
cd ../LightGBM/python-package && python3 setup.py install --precompile


