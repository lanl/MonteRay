#ifndef GPUTALLY_H_
#define GPUTALLY_H_

#include <string>

#include "MonteRayDefinitions.hh"

namespace MonteRay{

struct gpuTally {
	unsigned size;
	gpuTallyType_t* tally;
};

void ctor(struct gpuTally*);
void dtor(struct gpuTally*);
void copy(struct gpuTally* pCopy, struct gpuTally* pOrig);

void cudaDtor(gpuTally*);
void cudaCtor(gpuTally*, unsigned num);
void cudaCtor(gpuTally*, gpuTally*);

CUDA_CALLABLE_MEMBER
void score(struct gpuTally* ptr, unsigned cell, gpuTallyType_t value );

class gpuTallyHost {
public:

	gpuTallyHost(unsigned num);

    ~gpuTallyHost();

    void ctor( unsigned num);
    void dtor();

    void copyToGPU(void);
    void copyToCPU(void);

    gpuTallyType_t getTally(unsigned i ) const { return ptr->tally[i]; }
    void setTally(unsigned i, gpuTallyType_t value ) { ptr->tally[i] = value; }

    unsigned size() const { return ptr->size; }
    void clear(void);

    void write( std::string filename ) const;
    void read( std::string filename );

    gpuTally* getPtr(void) { return ptr; }

private:
    gpuTally* ptr;
    bool cudaCopyMade;

public:
    gpuTally* temp;
    gpuTally* ptr_device;

};

}

#endif /* GPUTALLY_H_ */
