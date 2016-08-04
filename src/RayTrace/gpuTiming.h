#ifndef GPUTIMING_H_
#define GPUTIMING_H_
#include "gpuGlobal.h"

namespace MonteRay{

struct gpuTiming {
	clock64_t start;
	clock64_t stop;
};

void ctor(struct gpuTiming*);
void dtor(struct gpuTiming*);
void copy(struct gpuTiming* pCopy, struct gpuTiming* pOrig);

class gpuTimingHost {
public:
	typedef gpuFloatType_t float_t;

	gpuTimingHost();

    ~gpuTimingHost();

    void copyToGPU(void);
    void copyToCPU(void);

    static clock64_t getCyclesPerSecond();
    void setRate( clock64_t Hz) { rate = Hz; }
    clock64_t getRate( void ) const { return rate; }

    void setClockStop( clock64_t clock){ ptr->stop = clock; }
    void setClockStart( clock64_t clock){ ptr->start = clock; }
    clock64_t getClockStop(){ return ptr->stop; }
    clock64_t getClockStart(){ return ptr->start; }
    double getGPUTime(void);

private:
    struct gpuTiming* ptr;
    bool cudaCopyMade;
    clock64_t rate; // cycles per second (Hz)

public:
    gpuTiming* ptr_device;

};

}

#endif /* GPUTIMING_H_ */
