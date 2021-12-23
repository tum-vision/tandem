# MVSNet C++ Wrapper

This folder contains the C++ wrapper for CVA-MVSNet using libtorch. It is also used for measuring inference speed, due to the better performance in comparison to the pure PyTorch version. The overall workflow consist of training a model in PyTorch, evaluating the model in PyTorch, exporting the model to TorchScript using PyTorch, and loading the TorchScript model with this C++ wrapper for fast inference. The loading step can optionally verify the loaded model against data saved from PyTorch, which is recommended. This code is under MIT license.

## Build
### Environment
For the paper we used CUDA 11.1, cuDNN 8.0.5, and `libtorch-1.9.0+cu111` with CUDA support and CXX11 ABI. However, we assume that the wrapper should work with a broad range of versions because it doesn't use version-specific features. We can sadly not offer a convenient installation script due to (a) different CUDA installation options and (b) the cuDNN download method that needs user input. You have to:

+ Install **CUDA** from [nvidia.com](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). Make sure that this doesn't interfere with other packages on your system and maybe ask your system administrator. CUDA 11.1 should be in your path, e.g. setting `CUDA_HOME`, `LD_LIBRARY_PATH` and `PATH` should be sufficient. You can also set the symlink `/usr/local/cuda`.

+ Install **cuDNN** from [nvidia.com](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html). Make sure to install a version that exactly matches your CUDA and PyTorch versions.
```
export TANDEM_CUDNN_LIBRARY=/path/to/cudnn/lib64
export TANDEM_CUDNN_INCLUDE_PATH=/path/to/cudnn/include
```

+ Install **LibTorch** from [pytorch.org](https://pytorch.org/get-started/locally/). For our exact version you can use
```
wget https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcu111.zip
export TANDEM_LIBTORCH_DIR=/path/to/unziped/libtorch
```


### Build and Test
CMake build in release mode and test together with downloaded data
```
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$TANDEM_LIBTORCH_DIR \
    -DCUDNN_LIBRARY=$TANDEM_CUDNN_LIBRARY \
    -DCUDNN_INCLUDE_PATH=$TANDEM_CUDNN_INCLUDE_PATH
make -j
make test
```
