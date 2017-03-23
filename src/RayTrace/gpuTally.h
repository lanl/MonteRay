#ifndef GPUTALLY_H_
#define GPUTALLY_H_

#include "MonteRayDefinitions.hh"

namespace MonteRay{

struct gpuTally {
	unsigned size;
	gpuTallyType_t* tally;
};

void ctor(struct gpuTally*);
void dtor(struct gpuTally*);
void copy(struct gpuTally* pCopy, struct gpuTally* pOrig);

#ifdef CUDA
void cudaDtor(gpuTally*);
void cudaCtor(gpuTally*, unsigned num);
void cudaCtor(gpuTally*, gpuTally*);
#endif

#ifdef CUDA
__device__
#endif
void score(struct gpuTally* ptr, unsigned cell, gpuTallyType_t value );

class gpuTallyHost {
public:

	gpuTallyHost(unsigned num);

    ~gpuTallyHost();

    void copyToGPU(void);
    void copyToCPU(void);

    gpuTallyType_t getTally(unsigned i ) const { return ptr->tally[i]; }
    void setTally(unsigned i, gpuTallyType_t value ) { ptr->tally[i] = value; }

    unsigned size() const { return ptr->size; }
    void clear(void);

private:
    gpuTally* ptr;
    gpuTally* temp;
    bool cudaCopyMade;

public:
    gpuTally* ptr_device;

};

}

#endif /* GPUTALLY_H_ */
