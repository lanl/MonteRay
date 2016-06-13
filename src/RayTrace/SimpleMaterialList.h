#ifndef SIMPLEMATERIALLIST_HH_
#define SIMPLEMATERIALLIST_HH_

#include "gpuGlobal.h"

#include "SimpleMaterial.h"

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
gpuFloatType_t getTotalXS(SimpleMaterialList* ptr, unsigned i, gpuFloatType_t E, gpuFloatType_t density);

#ifdef CUDA
__device__ __host__
#endif
unsigned materialIDtoIndex(SimpleMaterialList* ptr, unsigned id );

class SimpleMaterialListHost {
public:
    SimpleMaterialListHost(unsigned num);

    ~SimpleMaterialListHost();

    void copyToGPU(void);

    unsigned getNumberMaterials(void) const {
        return ::getNumberMaterials( pMatList );
    }

    unsigned getMaterialID(unsigned i) const {
        return ::getMaterialID( pMatList, i );
    }

    SimpleMaterial* getMaterial(unsigned i) const {
        return ::getMaterial( pMatList, i );
    }

    gpuFloatType_t getTotalXS(unsigned i, gpuFloatType_t E, gpuFloatType_t density) const {
        return ::getTotalXS( pMatList, i, E, density);
    }

    gpuFloatType_t launchGetTotalXS(unsigned i, gpuFloatType_t E, gpuFloatType_t density) const;

    unsigned materialIDtoIndex(unsigned id) const {
        return ::materialIDtoIndex( pMatList, id);
    }

    void add( unsigned i, SimpleMaterialHost& mat, unsigned id);
#ifndef CUDA
    void add( unsigned i, SimpleMaterial* mat, unsigned id);
#endif

    SimpleMaterialList* getPtr(void) const { return pMatList; }

    void write(std::ostream& outfile) const;
    void  read(std::istream& infile);

private:
    SimpleMaterialList* pMatList;
    SimpleMaterialList* temp;
    bool cudaCopyMade;

public:
    SimpleMaterialList* ptr_device;
    SimpleMaterial** material_device_ptr_list;

};

#endif /* SIMPLEMATERIALLIST_HH_ */
