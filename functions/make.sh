#!/usr/bin/env bash

# CUDA_PATH=/usr/local/cuda/

export CUDA_PATH=/usr/local/cuda/

python setup.py build_ext --inplace
rm -rf build

CUDA_ARCH="-gencode arch=compute_61,code=sm_61"

# clean build file
rm nms/src/*.o
rm psroi_pooling/src/cuda/*.o

# compile NMS
cd nms/src
echo "Compiling nms kernels by nvcc..."
nvcc -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH

cd ../
python build.py


# compile tf - psroi_pooling
cd ../
cd psroi_pooling/src/cuda
echo "Compiling psroi pooling kernels by nvcc..."
nvcc -c -o psroi_pooling.cu.o psroi_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../../
python build.py

# compile roi_align (based on Caffe2's implementation)
cd ../
cd roi_align/src
echo "Compiling roi align kernels by nvcc..."
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py


# compile roi_align (based on tensorflow's implementation)
echo "Compiling crop_and_resize kernels by nvcc..."
cd ../RoIAlign.pytorch/roi_align/src/cuda/
$CUDA_PATH/bin/nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_30 \
 -gencode=arch=compute_30,code=sm_30 \
 -gencode=arch=compute_50,code=sm_50 \
 -gencode=arch=compute_52,code=sm_52 \
 -gencode=arch=compute_60,code=sm_60 \
 -gencode=arch=compute_61,code=sm_61 \
 -gencode=arch=compute_62,code=sm_62 \

cd ../../../roi_align
python build.py

cd ..
python setup.py install

