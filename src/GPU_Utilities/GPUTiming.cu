#include "GPUTiming.hh"

#include "MonteRayMemory.hh"

namespace MonteRay {

gpuTimingHost::gpuTimingHost(){
     ptr = new gpuTiming;
     ctor(ptr);
     cudaCopyMade = false;
     ptr_device = NULL;

     rate = 0;
#ifdef __CUDACC__
     setRate( gpuTimingHost::getCyclesPerSecond() );
     copyToGPU();
#else
     setRate( CLOCKS_PER_SEC );
#endif
}

gpuTimingHost::~gpuTimingHost() {
    if( ptr != 0 ) {
        dtor( ptr );
        delete ptr;
        ptr = 0;
    }

#ifdef __CUDACC__
    if( cudaCopyMade ) {
        MonteRayDeviceFree( ptr_device );
    }
#endif
}

void
gpuTimingHost::copyToGPU(void){
#ifdef __CUDACC__
    if(cudaCopyMade != true ) {
        cudaCopyMade = true;

        // allocate target struct
        ptr_device = (gpuTiming*) MONTERAYDEVICEALLOC( sizeof( gpuTiming), std::string("GPUTiming::ptr_device") );
    }

    // copy data
    CUDA_CHECK_RETURN( cudaMemcpy(ptr_device, ptr, sizeof( gpuTiming ), cudaMemcpyHostToDevice));
#endif
}

void
gpuTimingHost::copyToCPU(void) {
#ifdef __CUDACC__
    cudaCopyMade = true;

    // copy data
    CUDA_CHECK_RETURN( cudaMemcpy(ptr, ptr_device, sizeof( gpuTiming ), cudaMemcpyDeviceToHost));
#endif
}

double
gpuTimingHost::getGPUTime(void){
    if( rate == 0 ) {
        throw std::runtime_error( "GPU rate not set." );
    }
#ifdef __CUDACC__
    if( !cudaCopyMade ){
        throw std::runtime_error( "gpuTiming not sent to GPU." );
    }
    copyToCPU();
#endif
    clock64_t deltaT;
    clock64_t start = ptr->start;
    clock64_t stop = ptr->stop;
    deltaT = stop - start;

    return double(deltaT) / rate;
}

clock64_t
gpuTimingHost::getCyclesPerSecond() {
    clock64_t Hz = 0;
    // Get device frequency in Hz
#ifdef __CUDACC__
    cudaDeviceProp prop;
    CUDA_CHECK_RETURN( cudaGetDeviceProperties(&prop, 0));
    Hz = clock64_t(prop.clockRate) * 1000;
#endif
    return Hz;
}

} // end namespace
