#ifndef GPUTIMING_H_
#define GPUTIMING_H_

#include <stdexcept>

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"

namespace MonteRay{

struct gpuTiming {
	clock64_t start;
	clock64_t stop;
};

inline void ctor(struct gpuTiming* pOrig) {
	pOrig->start = 0;
	pOrig->stop = 0;
}

inline void dtor(struct gpuTiming*){}

inline void copy(struct gpuTiming* pCopy, struct gpuTiming* pOrig) {
	pCopy->start = pOrig->start;
	pCopy->stop = pOrig->stop;
}

class gpuTimingHost {
public:
	typedef gpuFloatType_t float_t;

	gpuTimingHost(){
		ptr = new gpuTiming;
	    ctor(ptr);
	    cudaCopyMade = false;
	    ptr_device = NULL;

	    rate = 0;
#ifdef CUDA
	    setRate( gpuTimingHost::getCyclesPerSecond() );
	    copyToGPU();
#endif
	}

    ~gpuTimingHost() {
        if( ptr != 0 ) {
            dtor( ptr );
            delete ptr;
            ptr = 0;
        }

        if( cudaCopyMade ) {
#ifdef CUDA
        	cudaFree( ptr_device );
#endif
        }
    }

    void copyToGPU(void){
#ifdef CUDA
    	if(cudaCopyMade != true ) {
    		cudaCopyMade = true;

    		// allocate target struct
    		CUDA_CHECK_RETURN( cudaMalloc(&ptr_device, sizeof( gpuTiming) ));
    	}

    	// copy data
    	CUDA_CHECK_RETURN( cudaMemcpy(ptr_device, ptr, sizeof( gpuTiming ), cudaMemcpyHostToDevice));
#endif
    }

    void copyToCPU(void) {
#ifdef CUDA
    	cudaCopyMade = true;

    	// copy data
    	CUDA_CHECK_RETURN( cudaMemcpy(ptr, ptr_device, sizeof( gpuTiming ), cudaMemcpyDeviceToHost));
#endif
    }

    /// Returns number of cycles required for requested seconds
    static clock64_t getCyclesPerSecond() {
    	clock64_t Hz = 0;
        // Get device frequency in Hz
    #ifdef CUDA
        cudaDeviceProp prop;
        CUDA_CHECK_RETURN( cudaGetDeviceProperties(&prop, 0));
        Hz = clock64_t(prop.clockRate) * 1000;
    #endif
        return Hz;
    }

    void setRate( clock64_t Hz) { rate = Hz; }
    clock64_t getRate( void ) const { return rate; }

    void setClockStop( clock64_t clock){ ptr->stop = clock; }
    void setClockStart( clock64_t clock){ ptr->start = clock; }
    clock64_t getClockStop(){ return ptr->stop; }
    clock64_t getClockStart(){ return ptr->start; }

    double getGPUTime(void){
    	if( rate == 0 ) {
    		throw std::runtime_error( "GPU rate not set." );
    	}
    	if( !cudaCopyMade ){
    		throw std::runtime_error( "gpuTiming not sent to GPU." );
    	}
    	copyToCPU();

    	clock64_t deltaT;
    	clock64_t start = ptr->start;
    	clock64_t stop = ptr->stop;
    	deltaT = stop - start;

    	return double(deltaT) / rate;
    }

private:
    struct gpuTiming* ptr;
    bool cudaCopyMade;
    clock64_t rate; // cycles per second (Hz)

public:
    gpuTiming* ptr_device;

};

}

#endif /* GPUTIMING_H_ */
