#ifndef SIMPLEMATERIALLIST_HH_
#define SIMPLEMATERIALLIST_HH_

#include "MonteRayDefinitions.hh"

#include "SimpleMaterial.h"
#include "HashLookup.h"

namespace MonteRay{

struct SimpleMaterialList {
    unsigned numMaterials;
    unsigned* materialID;
    struct SimpleMaterial** materials;
};

void ctor(SimpleMaterialList* ptr, unsigned num );
void dtor(SimpleMaterialList* ptr);
void copy(SimpleMaterialList* pCopy, const SimpleMaterialList* const pOrig );

#ifdef CUDA
void cudaCtor(struct SimpleMaterialList*,struct SimpleMaterialList*);
void cudaDtor(struct SimpleMaterialList*);
#endif


#ifdef CUDA
__device__ __host__
#endif
unsigned getNumberMaterials(SimpleMaterialList* ptr);

#ifdef CUDA
__device__ __host__
#endif
unsigned getMaterialID(SimpleMaterialList* ptr, unsigned i );

#ifdef CUDA
__device__ __host__
#endif
SimpleMaterial* getMaterial(SimpleMaterialList* ptr, unsigned i );

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXS(SimpleMaterialList* ptr, unsigned i, HashLookup* pHash, unsigned hashBin, gpuFloatType_t E, gpuFloatType_t density);

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXS(SimpleMaterialList* ptr, unsigned i, gpuFloatType_t E, gpuFloatType_t density);

#ifdef CUDA
__device__ __host__
#endif
unsigned materialIDtoIndex(SimpleMaterialList* ptr, unsigned id );

class SimpleMaterialListHost {
public:
    SimpleMaterialListHost(unsigned numMaterials, unsigned maxNumIsotopes=20, unsigned nBins=8192);

    ~SimpleMaterialListHost();

    void copyToGPU(void);

    unsigned getNumberMaterials(void) const {
        return MonteRay::getNumberMaterials( pMatList );
    }

    unsigned getMaterialID(unsigned i) const {
        return MonteRay::getMaterialID( pMatList, i );
    }

    SimpleMaterial* getMaterial(unsigned i) const {
        return MonteRay::getMaterial( pMatList, i );
    }

    gpuFloatType_t getTotalXS(unsigned i, gpuFloatType_t E, gpuFloatType_t density) const {
    	unsigned index = pHash->getHashBin(E);
        return MonteRay::getTotalXS( pMatList, i, pHash->getPtr(), index, E, density);
    }

    gpuFloatType_t launchGetTotalXS(unsigned i, gpuFloatType_t E, gpuFloatType_t density) const;

    unsigned materialIDtoIndex(unsigned id) const {
        return MonteRay::materialIDtoIndex( pMatList, id);
    }

    void add( unsigned i, SimpleMaterialHost& mat, unsigned id);
#ifndef CUDA
    void add( unsigned i, SimpleMaterial* mat, unsigned id);
#endif

    MonteRay::SimpleMaterialList* getPtr(void) const { return pMatList; }
    MonteRay::HashLookupHost* getHashPtr(void) const { return pHash; }

    void write(std::ostream& outfile) const;
    void  read(std::istream& infile);

private:
    MonteRay::SimpleMaterialList* pMatList;
    MonteRay::SimpleMaterialList* temp;
    bool cudaCopyMade;

    HashLookupHost* pHash;

public:
    MonteRay::SimpleMaterialList* ptr_device;
    MonteRay::SimpleMaterial** material_device_ptr_list;

};

}

#endif /* SIMPLEMATERIALLIST_HH_ */
