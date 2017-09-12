#ifndef MONTERAYCROSSSECTION_HH_
#define MONTERAYCROSSSECTION_HH_

#include <iostream>
#include <vector>
#include <stdexcept>

#include "MonteRayDefinitions.hh"
#include "HashLookup.h"

namespace MonteRay{

struct MonteRayCrossSection {
	int id;
    unsigned numPoints;
    gpuFloatType_t AWR;
    gpuFloatType_t* energies;
    gpuFloatType_t* totalXS;

};

void ctor(struct MonteRayCrossSection*, unsigned num);
void dtor(struct MonteRayCrossSection*);
void copy(struct MonteRayCrossSection* pCopy, struct MonteRayCrossSection* pOrig );

void cudaDtor(MonteRayCrossSection*);
void cudaCtor(MonteRayCrossSection*, unsigned num);
void cudaCtor(MonteRayCrossSection*, MonteRayCrossSection*);

CUDA_CALLABLE_MEMBER
gpuFloatType_t getEnergy(const struct MonteRayCrossSection* pXS, unsigned i );

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXSByIndex(const struct MonteRayCrossSection* pXS, unsigned i );

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXS(const struct MonteRayCrossSection* pXS, gpuFloatType_t E );

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXS(const struct MonteRayCrossSection* pXS, const struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E );

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXSByIndex(const struct MonteRayCrossSection* pXS, unsigned i, gpuFloatType_t E );

CUDA_CALLABLE_MEMBER
unsigned getIndex(const struct MonteRayCrossSection* pXS, gpuFloatType_t E );

CUDA_CALLABLE_MEMBER
unsigned getIndex(const struct MonteRayCrossSection* pXS, const struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E );

CUDA_CALLABLE_MEMBER
unsigned getIndexBinary(const struct MonteRayCrossSection* pXS, unsigned lower, unsigned upper, gpuFloatType_t value );

CUDA_CALLABLE_MEMBER
unsigned getIndexLinear(const struct MonteRayCrossSection* pXS, unsigned lower, unsigned upper, gpuFloatType_t value );

CUDA_CALLABLE_MEMBER
gpuFloatType_t getAWR(const struct MonteRayCrossSection* pXS);

CUDA_CALLABLE_MEMBER
int getID(const struct MonteRayCrossSection* pXS);

CUDA_CALLABLE_MEMBER
void setID(struct MonteRayCrossSection* pXS, unsigned i);

CUDA_CALLABLE_KERNEL void kernelGetTotalXS(const struct MonteRayCrossSection* pXS, const HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t* result);

CUDA_CALLABLE_KERNEL void kernelGetTotalXS(const struct MonteRayCrossSection* pXS, gpuFloatType_t E, gpuFloatType_t* result);

//class ContinuousNeutron;

class MonteRayCrossSectionHost {
public:
    MonteRayCrossSectionHost(unsigned num);
    ~MonteRayCrossSectionHost();

    void copyToGPU(void);

    int getID(void) const { return MonteRay::getID( xs ); }
    void setID(unsigned id) { MonteRay::setID( xs, id ); }
    unsigned size(void) const { return xs->numPoints; }
    gpuFloatType_t getEnergy(unsigned i) const { return MonteRay::getEnergy(xs, i); }
    gpuFloatType_t getTotalXSByIndex(unsigned i) const { return MonteRay::getTotalXSByIndex(xs, i); }
    gpuFloatType_t getTotalXSByHashIndex(const struct HashLookup* pHash, unsigned i, gpuFloatType_t E) const;

    gpuFloatType_t getTotalXS( gpuFloatType_t E ) const { return MonteRay::getTotalXS(xs, E); }
    gpuFloatType_t getTotalXS( const struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E ) const;

    void setEnergy(unsigned i, gpuFloatType_t e) { xs->energies[i] = e; }
    unsigned getIndex( gpuFloatType_t e ) const { return MonteRay::getIndex( xs, e); }
    unsigned getIndex( const HashLookupHost* pHost, unsigned hashBin, gpuFloatType_t e ) const;

    void setTotalXS(unsigned i, gpuFloatType_t value) { xs->totalXS[i] = value; }

    void setTotalXS(unsigned i, gpuFloatType_t E, gpuFloatType_t value) {
    	if( i >= size() ) {
    		throw std::runtime_error( "Error: MonteRayCrossSectionHost::setTotalXS, invalid index");
    	}
        xs->energies[i] = E;
        xs->totalXS[i] = value;
    }

    gpuFloatType_t getAWR(void) const {return MonteRay::getAWR(xs); }
    void setAWR(gpuFloatType_t value) { xs->AWR = value; }

    void write(std::ostream& outfile) const;
    void  read(std::istream& infile);

    void write( const std::string& filename );
    void read( const std::string& filename );

    struct MonteRayCrossSection* getXSPtr(void) { return xs;}
    struct MonteRayCrossSection& getXSRef(void) { return *xs;}

    void load(struct MonteRayCrossSection* ptrXS );

    template<typename T>
    void load(const T& CrossSection ) {
    	typedef std::vector<double> xsec_t;

        unsigned num = CrossSection.getNumPoints();
        dtor( xs );
        ctor( xs, num );

        setAWR( CrossSection.getAWR() );

        xsec_t energies = CrossSection.getEnergyValues();
        xsec_t totalXS = CrossSection.getTotalValues();
        for( unsigned i=0; i<num; ++i ){
            xs->energies[i] = energies[i];
            xs->totalXS[i] = totalXS[i];
        }
    }

private:
    struct MonteRayCrossSection* xs;
    MonteRayCrossSection* temp;
    bool cudaCopyMade;

public:
    MonteRayCrossSection* xs_device;

};

gpuFloatType_t launchGetTotalXS( MonteRayCrossSectionHost* pXS, gpuFloatType_t energy);


}


#endif /* MONTERAYCROSSSECTION_HH_ */
