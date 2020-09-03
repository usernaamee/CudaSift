set -e
swig -python -c++ pycusift.i
g++ -std=c++14 -c -fPIC pycusift.cpp -I. `pkg-config --cflags opencv` `pkg-config --libs opencv` `pkg-config --libs cuda-10.2` `pkg-config --libs cudart-10.2`
g++ -std=c++14 -c -fPIC pycusift_wrap.cxx -I. `pkg-config --cflags opencv` `pkg-config --cflags python3` `pkg-config --libs opencv` `pkg-config --libs cuda-10.2` `pkg-config --libs cudart-10.2`
nvcc -c -shared -Xcompiler -fPIC  cudaImage.cu
nvcc -c -shared -Xcompiler -fPIC  cudaSiftH.cu
g++  -shared -static-libstdc++  -o _pycusift.so pycusift.o pycusift_wrap.o cudaImage.o cudaSiftH.o `pkg-config --libs opencv` `pkg-config --libs cuda-10.2` `pkg-config --libs cudart-10.2`
echo "Done!"
