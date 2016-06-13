FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

SET(CUDA_SEPARABLE_COMPILATION ON)
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}-DCUDA;-arch=sm_52 )
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -DCUDA; -Xcompiler -fPIC; -O3; -gencode arch=compute_32,code=sm_32; -ccbin gcc -std=c++11)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};${CUDA_DEBUG_FLAGS};-DCUDA; -arch=sm_52;-Xcompiler -fPIC -g;--relocatable-device-code=true;--cudart static)
message( STATUS "Using CUDA_NVCC_FLAGS=${CUDA_NVCC_FLAGS}")
