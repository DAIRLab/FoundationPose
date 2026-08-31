DIR=$(pwd)

# Build CUDA extensions for Ada (4090, sm_89) and Blackwell (50 series, sm_120).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;12.0}"

cd $DIR/mycpp/ && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j11
cd $DIR/bundlesdf/mycuda && rm -rf build *egg* && pip install -e .

cd ${DIR}
