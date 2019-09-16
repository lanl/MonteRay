#ifndef MONTERAYMATERIAL_HH_
#define MONTERAYMATERIAL_HH_

#include <ostream>
#include <istream>
#include <vector>

#include "MonteRayTypes.hh"

namespace MonteRay{

class MonteRayCrossSection;
class MonteRayCrossSectionHost;
class HashLookup;
class HashLookupHost;

struct MonteRayMaterial {
    unsigned numIsotopes;

    gpuFloatType_t AtomicWeight;

    gpuFloatType_t* fraction;
    struct MonteRayCrossSection** xs;
};

void ctor(struct MonteRayMaterial*, unsigned numIsotopes);
void dtor(struct MonteRayMaterial*);
void copy(struct MonteRayMaterial* pCopy, struct MonteRayMaterial* pOrig);

void cudaCtor(struct MonteRayMaterial*,struct MonteRayMaterial*);
void cudaDtor(struct MonteRayMaterial*);

CUDA_CALLABLE_MEMBER
unsigned getNumIsotopes(struct MonteRayMaterial* ptr );

CUDA_CALLABLE_KERNEL  kernelGetNumIsotopes(MonteRayMaterial* pMat, unsigned* results);

CUDA_CALLABLE_MEMBER
gpuFloatType_t getFraction(struct MonteRayMaterial* ptr, unsigned i);

CUDA_CALLABLE_MEMBER
void normalizeFractions(struct MonteRayMaterial* ptr );

CUDA_CALLABLE_MEMBER
void calcAtomicWeight(struct MonteRayMaterial* ptr );

CUDA_CALLABLE_MEMBER
gpuFloatType_t getAtomicWeight(struct MonteRayMaterial* ptr );

CUDA_CALLABLE_MEMBER
gpuFloatType_t getMicroTotalXS(const struct MonteRayMaterial* ptr, const HashLookup* pHash, unsigned HashBin, gpuFloatType_t E);

CUDA_CALLABLE_MEMBER
gpuFloatType_t getMicroTotalXS(const struct MonteRayMaterial* ptr, gpuFloatType_t E);

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXS(const struct MonteRayMaterial* ptr, const HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t density);

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXS(const struct MonteRayMaterial* ptr, gpuFloatType_t E, gpuFloatType_t density);

CUDA_CALLABLE_MEMBER
void setID(struct MonteRayMaterial* ptr, unsigned index, unsigned id );

CUDA_CALLABLE_MEMBER
int getID(struct MonteRayMaterial* ptr, unsigned index );

CUDA_CALLABLE_MEMBER
void cudaAdd(struct MonteRayMaterial* ptr, struct MonteRayCrossSection* xs, unsigned index );

class MonteRayMaterialHost {
public:
    typedef gpuFloatType_t float_t;

    MonteRayMaterialHost(unsigned numIsotopes);

    ~MonteRayMaterialHost();

    void copyToGPU(void);
    void copyOwnedCrossSectionsToGPU(void);

    unsigned getNumIsotopes(void) const {
        return MonteRay::getNumIsotopes( pMat );
    }

    unsigned launchGetNumIsotopes(void);

    gpuFloatType_t getFraction(unsigned i) const {
        return MonteRay::getFraction(pMat, i);
    }

    gpuFloatType_t getMicroTotalXS(HashLookup* pHash, gpuFloatType_t E );

    gpuFloatType_t getTotalXS(gpuFloatType_t E, gpuFloatType_t density=1.0 ){
        return MonteRay::getTotalXS(pMat, E, density);
    }

    gpuFloatType_t getTotalXS(HashLookup* pHash, gpuFloatType_t E, gpuFloatType_t density );

    gpuFloatType_t launchGetTotalXS(gpuFloatType_t E, gpuFloatType_t density=1.0 );
    gpuFloatType_t launchGetTotalXSViaHash(HashLookupHost& hash, gpuFloatType_t E, gpuFloatType_t density );

    void add(unsigned index, MonteRayCrossSectionHost& xs, gpuFloatType_t frac );
    void add(unsigned index, struct MonteRayCrossSection* xs, gpuFloatType_t frac );

    void setID( unsigned index, unsigned id);
    int getID( unsigned index );

    void normalizeFractions(void) { MonteRay::normalizeFractions(pMat); }
    void calcAWR(void){ MonteRay::calcAtomicWeight(pMat); }

    gpuFloatType_t getAtomicWeight(void) const { return MonteRay::getAtomicWeight(pMat); }


    void write(std::ostream& outfile) const;
    void  read(std::istream& infile, HashLookupHost* pHash = nullptr);
    void writeToFile( const std::string& filename) const;
    void readFromFile( const std::string& filename, HashLookupHost* pHash = nullptr);

    void load(MonteRayMaterial* ptrMat );

    struct MonteRayMaterial* getPtr(void) { return pMat; }

private:
    struct MonteRayMaterial* pMat = nullptr;
    MonteRayMaterial* temp = nullptr;
    bool cudaCopyMade = false;

    std::vector<MonteRayCrossSectionHost*> ownedCrossSections;

public:
    MonteRayMaterial* ptr_device = nullptr;
    MonteRayCrossSection** isotope_device_ptr_list = nullptr;

};


} // end namespace MonteRay

#include "Material.hh"

#endif /* MONTERAYMATERIAL_HH_ */
