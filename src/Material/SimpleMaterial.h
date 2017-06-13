#ifndef SIMPLEMATERIAL_HH_
#define SIMPLEMATERIAL_HH_

#include "MonteRayDefinitions.hh"
#include "MonteRayCrossSection.hh"

namespace MonteRay{

struct SimpleMaterial {
    unsigned numIsotopes;

    gpuFloatType_t AtomicWeight;

    gpuFloatType_t* fraction;
    struct MonteRayCrossSection** xs;
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
gpuFloatType_t getMicroTotalXS(struct SimpleMaterial* ptr, HashLookup* pHash, unsigned HashBin, gpuFloatType_t E);

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getMicroTotalXS(struct SimpleMaterial* ptr, gpuFloatType_t E);

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXS(struct SimpleMaterial* ptr, HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t density);

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXS(struct SimpleMaterial* ptr, gpuFloatType_t E, gpuFloatType_t density);

#ifdef CUDA
__device__ __host__
#endif
void setID(struct SimpleMaterial* ptr, unsigned index, unsigned id );

#ifdef CUDA
__device__ __host__
#endif
int getID(struct SimpleMaterial* ptr, unsigned index );

#ifdef CUDA
__device__ __host__
#endif
void cudaAdd(struct SimpleMaterial* ptr, struct MonteRayCrossSection* xs, unsigned index );

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

    gpuFloatType_t getMicroTotalXS(HashLookup* pHash, gpuFloatType_t E ){
    	unsigned HashBin = getHashBin(pHash, E);
        return MonteRay::getMicroTotalXS(pMat, pHash, HashBin, E);
    }

    gpuFloatType_t getTotalXS(HashLookup* pHash, gpuFloatType_t E, gpuFloatType_t density ){
    	 unsigned HashBin = getHashBin(pHash, E);
         return MonteRay::getTotalXS(pMat, pHash, HashBin, E, density);
     }

    void add(unsigned index, MonteRayCrossSectionHost& xs, gpuFloatType_t frac );
#ifndef CUDA
    void add(unsigned index, struct MonteRayCrossSection* xs, gpuFloatType_t frac );
#endif

    void setID( unsigned index, unsigned id);
    int getID( unsigned index );

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
    MonteRayCrossSection** isotope_device_ptr_list;

};

}

#endif /* SIMPLEMATERIAL_HH_ */
