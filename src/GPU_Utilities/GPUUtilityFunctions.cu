#include <cstdio>
#include <unistd.h>
#include <iostream>

#include "GPUErrorCheck.hh"
#include "GPUUtilityFunctions.hh"
#include "MonteRayDefinitions.hh"

namespace MonteRay{

void cudaReset(const bool verbose) {
#ifdef __CUDACC__
    char hostname[1024];
    gethostname(hostname,1024);

    int deviceID = getCudaDevice(verbose);

    if(verbose) std::cout << "MonteRay::cudaReset -- " << hostname << ", device " << deviceID <<  " -- Reseting the GPU.\n";
    cudaError_t error = cudaDeviceReset();
    if( error != cudaSuccess ) {
        std::cout << "MonteRay::cudaReset -- " << hostname << ", device=" << deviceID << " -- cudaDeviceReset() call failed.\n";
        throw std::runtime_error ("MonteRay::cudaReset -- call to cudaDeviceReset() failed.");
    }
#endif
}

void gpuReset(const bool verbose) {
    cudaReset(verbose);
}

void gpuCheck(const bool verbose) {
#ifdef __CUDACC__
    int deviceCount;
    char hostname[1024];
    gethostname(hostname,1024);

    if(verbose) std::cout << "MonteRay::gpuCheck -- " << hostname << " -- Initializing and checking GPU Status.\n";

    CUresult result_error = cuInit(0);
    if( result_error != CUDA_SUCCESS ) {
        std::cout << "MonteRay::gpuCheck -- " << hostname << " -- cuInit(0) call failed.\n";
        throw std::runtime_error ("MonteRay::gpuCheck -- call to cuInit(0) failed.");
    }

    result_error = cuDeviceGetCount(&deviceCount);
    if( result_error != CUDA_SUCCESS ) {
        std::cout << "MonteRay::gpuCheck -- " << hostname << " -- cuDeviceGetCount() call failed.\n";
        throw std::runtime_error ("MonteRay::gpuCheck -- call to cuDeviceGetCount() failed.");
    } else {
        if(verbose) std::cout << "MonteRay::gpuCheck -- " << hostname << " -- cuDeviceGetCount() reported " << deviceCount << "GPU(s) on the host. \n";
    }
#endif
}

void gpuCheck() { gpuCheck(false); }

void gpuInfo() {
#ifdef __CUDACC__
    int deviceCount;

    CUresult result_error = cuDeviceGetCount(&deviceCount);
    if( result_error != CUDA_SUCCESS ) {
        printf("CUDA call: cuDeviceGetCount failed!\n");
        exit(1);
    }

    printf("Number of CUDA devices=%d\n",deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Compute capability %d,%d\n", prop.major, prop.minor);
        printf("  Memory Clock Rate (KHz): %d\n",
                prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
                prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
                2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
#endif
}

int getNumberOfGPUS(const bool verbose) {
    int count = 0;
    char hostname[1024];
    gethostname(hostname,1024);

    if(verbose) std::cout << "MonteRay::getNumberOfGPUS -- " << hostname << " -- Getting number of GPUs on the host.\n";

#ifdef __CUDACC__
    cudaError_t error = cudaGetDeviceCount( &count ) ;
    if(verbose) std::cout << "MonteRay::getNumberOfGPUS -- " << hostname << " -- Number of GPUs = " << count << ".\n";
    if( error != cudaSuccess ) {
        std::cout << "MonteRay::getNumberOfGPUS -- " << hostname << " -- getNumberOfGPUS() call failed.\n";
        throw std::runtime_error ("MonteRay::getNumberOfGPUS -- call to cudaGetDeviceCount() failed.");
    }

#endif
    return count;
}

void setCudaDevice(int id, const bool verbose ) {
    char hostname[1024];
    gethostname(hostname,1024);

    if(verbose) std::cout << "MonteRay::setCudaDevice -- " << hostname << " -- setting the cuda device, requested device id = " << id << "\n";

#ifdef __CUDACC__
    cudaError_t error = cudaSetDevice( id ) ;
    if( error != cudaSuccess ) {
        std::cout << "MonteRay::setCudaDevice -- " << hostname << " -- cudaSetDevice() call failed or device id = " << id << ".\n";
        throw std::runtime_error ("MonteRay::setCudaDevice -- call to cudaSetDevice() failed.");
    }
#endif
}

void setCudaPrintBufferSize( size_t size, const bool verbose ) {
    char hostname[1024];
    gethostname(hostname,1024);

#ifdef __CUDACC__
    int device = getCudaDevice(verbose);
    cudaError_t error = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size );
    if( error != cudaSuccess ) {
        std::cout << "MonteRay::setCudaPrintBufferSize -- " << hostname << ", device=" << device << " -- cudaDeviceSetLimit() call failed.\n";
        throw std::runtime_error ("MonteRay::setCudaPrintBufferSize -- call to cudaDeviceSetLimit() failed.");
    }
#endif
}

int getCudaDevice( const bool verbose ) {
    int deviceID = 0;
    char hostname[1024];
    gethostname(hostname,1024);

    if(verbose) std::cout << "MonteRay::getCudaDevice -- " << hostname << " -- getting the current cuda device id...\n";
#ifdef __CUDACC__
    cudaError_t error = cudaGetDevice(&deviceID);
    if( error != cudaSuccess ) {
        std::cout << "MonteRay::getCudaDevice -- " << hostname << " -- cudaGetDevice() call failed.\n";
        throw std::runtime_error ("MonteRay::getCudaDevice -- call to cudaGetDevice() failed.");
    }
#endif
    return deviceID;
}

void setCudaStackSize( size_t size, const bool verbose) {
    char hostname[1024];
    gethostname(hostname,1024);
    int device;

    if(verbose) std::cout << "MonteRay::setCudaDevice -- " << hostname << " -- setting the cuda stack size...\n";

#ifdef __CUDACC__
    device = getCudaDevice(verbose);

    if(verbose) std::cout << "MonteRay::setCudaDevice -- " << hostname << " -- setting the cuda stack size for device =" << device << " ...\n";
    cudaError_t error = cudaDeviceSetLimit( cudaLimitStackSize, size );
    if( error != cudaSuccess ) {
        std::cout << "MonteRay::setCudaStackSize -- " << hostname << ", device=" << device << " -- cudaSetDevice() call failed.\n";
        throw std::runtime_error ("MonteRay::setCudaStackSize -- call to cudaSetDevice() failed.");
    }
#endif
}

} /* namespace MonteRay */
