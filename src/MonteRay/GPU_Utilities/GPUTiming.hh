#ifndef GPUTIMING_H_
#define GPUTIMING_H_

#include <stdexcept>

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"

#ifndef __CUDACC__
#include <ctime>
#endif

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

    gpuTimingHost();

    ~gpuTimingHost();

    void copyToGPU(void);

    void copyToCPU(void);

    /// Returns number of cycles required for requested seconds
    static clock64_t getCyclesPerSecond();

    void setRate( clock64_t Hz) { rate = Hz; }
    clock64_t getRate( void ) const { return rate; }

    void setClockStop( clock64_t clock){ ptr->stop = clock; }
    void setClockStart( clock64_t clock){ ptr->start = clock; }
    clock64_t getClockStop(){ return ptr->stop; }
    clock64_t getClockStart(){ return ptr->start; }

    double getGPUTime(void);

    gpuTiming* getPtr(void) const { return ptr; }

private:
    struct gpuTiming* ptr;
    bool cudaCopyMade;
    clock64_t rate; // cycles per second (Hz)

public:
    gpuTiming* ptr_device;

};

}

#endif /* GPUTIMING_H_ */
