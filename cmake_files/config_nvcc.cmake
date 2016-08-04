FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

SET(CUDA_SEPARABLE_COMPILATION ON)
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}-DCUDA;-arch=sm_52 )
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -DCUDA; -Xcompiler -fPIC; -O3; -gencode arch=compute_32,code=sm_32; -ccbin gcc -std=c++11)
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};${CUDA_DEBUG_FLAGS};-DCUDA; -arch=sm_52;-O3; -Xcompiler -fPIC -g;--relocatable-device-code=true;--cudart static)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};${CUDA_DEBUG_FLAGS};-DCUDA; -arch=sm_52; -O3; --use_fast_math; -Xcompiler -fPIC;--relocatable-device-code=true;--cudart static)
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};${CUDA_DEBUG_FLAGS};-DCUDA; -G; -g; -sp-bound-check; --opt-level=0; --prec-div=true; --prec-sqrt=true; --ftz=false; --fmad=false; -arch=sm_35;-Xcompiler -fPIC -g;--relocatable-device-code=true;--cudart static)
message( STATUS "Using CUDA_NVCC_FLAGS=${CUDA_NVCC_FLAGS}")
