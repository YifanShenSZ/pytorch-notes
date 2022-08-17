# Installation

## linux
On a ubuntu machine, `python setup.py install` invokes `cmake` under the hood, so the controlling environmental variables should be in `cmake` format:
* `export USE_CUDA=0` should be `export USE_CUDA=OFF`
* `export CC=g++` should be `export CMAKE_C_COMPILER=gcc; export CMAKE_CXX_COMPILER=g++`

Unfortunately, we cannot use intel compiler because icc does not support `__buildin_shuffle`; intel have their own version with different semantics

## macOS
For now mac is easier, since:
* it has no NVIDIA GPU at all, so no need to manually disable cuda
* guess clang is the only available main-stream compiler?

Maybe things will change with MPS support coming in, we will update then
