#ifndef SIMPLEMATERIAL_HH_
#define SIMPLEMATERIAL_HH_

#include "gpuGlobal.h"
#include "SimpleCrossSection.h"

namespace MonteRay{

struct SimpleMaterial {
    unsigned numIsotopes;

    gpuFloatType_t AtomicWeight;

    gpuFloatType_t* fraction;
    struct SimpleCrossSection** xs;
};

void ctor(struct SimpleMaterial*, unsigned numIsotopes);
void dtor(struct SimpleMaterial*);
void copy(struct SimpleMaterial* pCopy, struct SimpleMaterial* pOrig);

#ifdef CUDA
void cudaCtor(struct SimpleMaterial*,struct SimpleMaterial*);
void cudaDtor(struct SimpleMaterial*);
#endif

#ifdef CUDA
__device__ __host__
#endif
unsigned getNumIsotopes(struct SimpleMaterial* ptr );

#ifdef CUDA
__global__ void kernelGetNumIsotopes(SimpleMaterial* pMat, unsigned* results);
#endif

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getFraction(struct SimpleMaterial* ptr, unsigned i);

#ifdef CUDA
__device__ __host__
#endif
void normalizeFractions(struct SimpleMaterial* ptr );

#ifdef CUDA
__device__ __host__
#endif
void calcAtomicWeight(struct SimpleMaterial* ptr );

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getAtomicWeight(struct SimpleMaterial* ptr );

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getMicroTotalXS(struct SimpleMaterial* ptr, gpuFloatType_t E);

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXS(struct SimpleMaterial* ptr, gpuFloatType_t E, gpuFloatType_t density);

#ifdef CUDA
__device__ __host__
#endif
void cudaAdd(struct SimpleMaterial* ptr, struct SimpleCrossSection* xs, unsigned index );

class SimpleMaterialHost {
public:
	typedef gpuFloatType_t float_t;

    SimpleMaterialHost(unsigned numIsotopes);

    ~SimpleMaterialHost();

    void copyToGPU(void);

    unsigned getNumIsotopes(void) const {
        return MonteRay::getNumIsotopes( pMat );
    }

    unsigned launchGetNumIsotopes(void);

    gpuFloatType_t getFraction(unsigned i) const {
        return MonteRay::getFraction(pMat, i);
    }

    gpuFloatType_t getMicroTotalXS(gpuFloatType_t E ){
        return MonteRay::getMicroTotalXS(pMat, E);
    }

    gpuFloatType_t getTotalXS(gpuFloatType_t E, gpuFloatType_t density ){
         return MonteRay::getTotalXS(pMat, E, density);
     }

    void add(unsigned index, SimpleCrossSectionHost& xs, gpuFloatType_t frac );
#ifndef CUDA
    void add(unsigned index, struct SimpleCrossSection* xs, gpuFloatType_t frac );
#endif

    void normalizeFractions(void) { MonteRay::normalizeFractions(pMat); }
    void calcAWR(void){ MonteRay::calcAtomicWeight(pMat); }

    gpuFloatType_t getAtomicWeight(void) const { return MonteRay::getAtomicWeight(pMat); }

    void write(std::ostream& outfile) const;
    void  read(std::istream& infile);

    void load(SimpleMaterial* ptrMat );

    struct SimpleMaterial* getPtr(void) { return pMat; }

private:
    struct SimpleMaterial* pMat;
    SimpleMaterial* temp;
    bool cudaCopyMade;

public:
    SimpleMaterial* ptr_device;
    SimpleCrossSection** isotope_device_ptr_list;

};

}

#endif /* SIMPLEMATERIAL_HH_ */
