#ifndef MONTERAYMATERIALLIST_HH_
#define MONTERAYMATERIALLIST_HH_

#include <sstream>

#include "MonteRayConstants.hh"

namespace MonteRay{

class HashLookup;
class HashLookupHost;
class MonteRayMaterial;
class MonteRayMaterialHost;

struct MonteRayMaterialList {
    unsigned numMaterials;
    unsigned* materialID;
    struct MonteRayMaterial** materials;
};

void ctor(MonteRayMaterialList* ptr, unsigned num );
void dtor(MonteRayMaterialList* ptr);
void copy(MonteRayMaterialList* pCopy, const MonteRayMaterialList* const pOrig );

void cudaCtor(struct MonteRayMaterialList*,struct MonteRayMaterialList*);
void cudaDtor(struct MonteRayMaterialList*);

CUDA_CALLABLE_MEMBER
unsigned getNumberMaterials(MonteRayMaterialList* ptr);

CUDA_CALLABLE_MEMBER
unsigned getMaterialID(MonteRayMaterialList* ptr, unsigned i );

CUDA_CALLABLE_MEMBER
MonteRayMaterial* getMaterial(MonteRayMaterialList* ptr, unsigned i );

CUDA_CALLABLE_MEMBER
const MonteRayMaterial* getMaterial(const MonteRayMaterialList* ptr, unsigned i );

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXS(const MonteRayMaterialList* ptr, unsigned i, const HashLookup* pHash, unsigned hashBin, gpuFloatType_t E, gpuFloatType_t density);

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXS(const MonteRayMaterialList* ptr, unsigned i, gpuFloatType_t E, gpuFloatType_t density);

CUDA_CALLABLE_MEMBER
unsigned materialIDtoIndex(MonteRayMaterialList* ptr, unsigned id );

class MonteRayMaterialListHost {
public:
    MonteRayMaterialListHost(unsigned numMaterials, unsigned maxNumIsotopes=20, unsigned nBins=8192);

    ~MonteRayMaterialListHost();

    void copyToGPU(void);

    unsigned getNumberMaterials(void) const {
        return MonteRay::getNumberMaterials( pMatList );
    }

    unsigned getMaterialID(unsigned i) const {
        return MonteRay::getMaterialID( pMatList, i );
    }

    MonteRayMaterial* getMaterial(unsigned i) const {
        return MonteRay::getMaterial( pMatList, i );
    }

    gpuFloatType_t getTotalXS(unsigned i, gpuFloatType_t E, gpuFloatType_t density, ParticleType_t ParticleType = neutron) const;

    gpuFloatType_t launchGetTotalXS(unsigned i, gpuFloatType_t E, gpuFloatType_t density) const;

    unsigned materialIDtoIndex(unsigned id) const;

    void add( unsigned i, MonteRayMaterialHost& mat, unsigned id);
    void add( unsigned i, MonteRayMaterial* mat, unsigned id);

    const MonteRay::MonteRayMaterialList* getPtr(void) const { return pMatList; }
    const MonteRay::HashLookupHost* getHashPtr(void) const { return pHash; }

    void write(std::ostream& outfile) const;
    void  read(std::istream& infile);

private:
    MonteRay::MonteRayMaterialList* pMatList;
    MonteRay::MonteRayMaterialList* temp;
    bool cudaCopyMade;

    HashLookupHost* pHash;

public:
    MonteRay::MonteRayMaterialList* ptr_device;
    MonteRay::MonteRayMaterial** material_device_ptr_list;

};

}

#endif /* MONTERAYMATERIALLIST_HH_ */
