#ifndef HASHLOOKUP_HH_
#define HASHLOOKUP_HH_

#include <vector>

#include "MonteRayDefinitions.hh"

namespace MonteRay{

class MonteRayCrossSectionHost;
class MonteRayCrossSection;

struct HashLookup {
	unsigned maxNumIsotopes;
    unsigned numIsotopes;
    unsigned N;
    gpuFloatType_t eMin;
    gpuFloatType_t eMax;
    gpuFloatType_t delta;
    unsigned* binBounds;
};

void ctor(HashLookup* ptr, unsigned num, unsigned nBins=8000 );
void dtor(HashLookup* ptr);
void copy(HashLookup* pCopy, const HashLookup* const pOrig );

void cudaCtor(struct HashLookup*,struct HashLookup*);
void cudaDtor(struct HashLookup*);

CUDA_CALLABLE_MEMBER
unsigned getMaxNumIsotopes(const HashLookup* ptr);

CUDA_CALLABLE_MEMBER
unsigned getNumIsotopes(const HashLookup* ptr);

CUDA_CALLABLE_MEMBER
bool setHashMinMax(HashLookup* ptr, MonteRayCrossSection* xs );

CUDA_CALLABLE_MEMBER
void setHashBinBounds(HashLookup* ptr, MonteRayCrossSection* xs, unsigned j );

CUDA_CALLABLE_MEMBER
unsigned getNBins(const HashLookup* ptr );

CUDA_CALLABLE_MEMBER
gpuFloatType_t getMaxEnergy(const HashLookup* ptr );

CUDA_CALLABLE_MEMBER
gpuFloatType_t getMinEnergy(const HashLookup* ptr );

CUDA_CALLABLE_MEMBER
unsigned getHashBin(const HashLookup* ptr, gpuFloatType_t energy );

CUDA_CALLABLE_MEMBER
unsigned getLowerBoundbyIndex(const HashLookup* ptr, unsigned isotope, unsigned index );

CUDA_CALLABLE_MEMBER
unsigned getUpperBoundbyIndex(const HashLookup* ptr, unsigned isotope, unsigned index );

CUDA_CALLABLE_MEMBER
unsigned getBinBoundIndex(const HashLookup* ptr, unsigned isotope, unsigned index );

class HashLookupHost {
public:
    HashLookupHost(unsigned num, unsigned nBins=8192);

    ~HashLookupHost();

    void copyToGPU(void);


    unsigned getMaxNumIsotopes(void) const {
        return MonteRay::getMaxNumIsotopes( ptr );
    }

    unsigned getNumIsotopes(void) const {
        return MonteRay::getNumIsotopes( ptr );
    }

    gpuFloatType_t getMaxEnergy(void) const {
    	return MonteRay::getMaxEnergy( ptr );
    }

    gpuFloatType_t getMinEnergy(void) const {
    	return MonteRay::getMinEnergy( ptr );
    }

    unsigned getHashBin( gpuFloatType_t energy) const {
    	return MonteRay::getHashBin( ptr, energy );
    }

    void addIsotope( MonteRayCrossSectionHost* xs );
    void addIsotope( MonteRayCrossSection* xs );

    unsigned getNBins(void);

    unsigned getLowerBoundbyIndex( unsigned isotope, unsigned index) const;
    unsigned getUpperBoundbyIndex( unsigned isotope, unsigned index) const;

    unsigned getBinBoundIndex( unsigned isotope, unsigned index) const{
    	return MonteRay::getBinBoundIndex(ptr, isotope, index );
    }

//    unsigned getMaterialID(unsigned i) const {
//        return MonteRay::getMaterialID( pMatList, i );
//    }
//
//    MonteRayMaterial* getMaterial(unsigned i) const {
//        return MonteRay::getMaterial( pMatList, i );
//    }
//
//    gpuFloatType_t getTotalXS(unsigned i, gpuFloatType_t E, gpuFloatType_t density) const {
//        return MonteRay::getTotalXS( pMatList, i, E, density);
//    }
//
//    gpuFloatType_t launchGetTotalXS(unsigned i, gpuFloatType_t E, gpuFloatType_t density) const;
//
//    unsigned materialIDtoIndex(unsigned id) const {
//        return MonteRay::materialIDtoIndex( pMatList, id);
//    }
//
//    void add( unsigned i, MonteRayMaterialHost& mat, unsigned id);
//#ifndef __CUDACC__
//    void add( unsigned i, MonteRayMaterial* mat, unsigned id);
//#endif
//
    const MonteRay::HashLookup* getPtr(void) const { return ptr; }
    const MonteRay::HashLookup* getPtrDevice(void) const { return ptr_device; }
//
//    void write(std::ostream& outfile) const;
//    void  read(std::istream& infile);

private:
    MonteRay::HashLookup* ptr;
    MonteRay::HashLookup* temp;
    bool cudaCopyMade;

    std::vector<MonteRay::MonteRayCrossSection*> xsList;

public:
    MonteRay::HashLookup* ptr_device;

};

}

#endif /* HashLookup_HH_ */
