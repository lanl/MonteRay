#ifndef SIMPLECROSSSECTION_HH_
#define SIMPLECROSSSECTION_HH_

#include <iostream>
#include <vector>
#include <stdexcept>

#include "MonteRayDefinitions.hh"
#include "HashLookup.h"

namespace MonteRay{

struct SimpleCrossSection {
	int id;
    unsigned numPoints;
    gpuFloatType_t AWR;
    gpuFloatType_t* energies;
    gpuFloatType_t* totalXS;

};

void ctor(struct SimpleCrossSection*, unsigned num);
void dtor(struct SimpleCrossSection*);
void copy(struct SimpleCrossSection* pCopy, struct SimpleCrossSection* pOrig );

#ifdef CUDA
void cudaDtor(SimpleCrossSection*);
void cudaCtor(SimpleCrossSection*, unsigned num);
void cudaCtor(SimpleCrossSection*, SimpleCrossSection*);
#endif

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getEnergy(struct SimpleCrossSection* pXS, unsigned i );

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXSByIndex(struct SimpleCrossSection* pXS, unsigned i );

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXS(struct SimpleCrossSection* pXS, gpuFloatType_t E );

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXS(struct SimpleCrossSection* pXS, struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E );

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getTotalXSByIndex(struct SimpleCrossSection* pXS, unsigned i, gpuFloatType_t E );

#ifdef CUDA
__device__ __host__
#endif
unsigned getIndex(struct SimpleCrossSection* pXS, gpuFloatType_t E );

#ifdef CUDA
__device__ __host__
#endif
unsigned getIndex(struct SimpleCrossSection* pXS, struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E );


#ifdef CUDA
__device__ __host__
#endif
unsigned getIndexBinary(struct SimpleCrossSection* pXS, unsigned lower, unsigned upper, gpuFloatType_t value );

#ifdef CUDA
__device__ __host__
#endif
unsigned getIndexLinear(struct SimpleCrossSection* pXS, unsigned lower, unsigned upper, gpuFloatType_t value );

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getAWR(struct SimpleCrossSection* pXS);

#ifdef CUDA
__device__ __host__
#endif
int getID(struct SimpleCrossSection* pXS);

#ifdef CUDA
__device__ __host__
#endif
void setID(struct SimpleCrossSection* pXS, unsigned i);

#ifdef CUDA
__global__ void kernelGetTotalXS(struct SimpleCrossSection* pXS, HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t* result);
#endif

#ifdef CUDA
__global__ void kernelGetTotalXS(struct SimpleCrossSection* pXS, gpuFloatType_t E, gpuFloatType_t* result);
#endif

class ContinuousNeutron;

class SimpleCrossSectionHost {
public:
    SimpleCrossSectionHost(unsigned num);
    ~SimpleCrossSectionHost();

    void copyToGPU(void);

    int getID(void) const { return MonteRay::getID( xs ); }
    void setID(unsigned id) { MonteRay::setID( xs, id ); }
    unsigned size(void) const { return xs->numPoints; }
    gpuFloatType_t getEnergy(unsigned i) const { return MonteRay::getEnergy(xs, i); }
    gpuFloatType_t getTotalXSByIndex(unsigned i) const { return MonteRay::getTotalXSByIndex(xs, i); }
    gpuFloatType_t getTotalXSByHashIndex(struct HashLookup* pHash, unsigned i, gpuFloatType_t E) const;

    gpuFloatType_t getTotalXS( gpuFloatType_t E ) const { return MonteRay::getTotalXS(xs, E); }
    gpuFloatType_t getTotalXS( struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E ) const;

    void setEnergy(unsigned i, gpuFloatType_t e) { xs->energies[i] = e; }
    unsigned getIndex( gpuFloatType_t e ) const { return MonteRay::getIndex( xs, e); }
    unsigned getIndex( HashLookupHost* pHost, unsigned hashBin, gpuFloatType_t e ) const;

    void setTotalXS(unsigned i, gpuFloatType_t value) { xs->totalXS[i] = value; }

    void setTotalXS(unsigned i, gpuFloatType_t E, gpuFloatType_t value) {
    	if( i >= size() ) {
    		throw std::runtime_error( "Error: SimpleCrossSectionHost::setTotalXS, invalid index");
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

    struct SimpleCrossSection* getXSPtr(void) { return xs;}
    struct SimpleCrossSection& getXSRef(void) { return *xs;}


#if !defined( CUDA )
    void load( const ContinuousNeutron& cn );
#endif

    void load(struct SimpleCrossSection* ptrXS );

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
    struct SimpleCrossSection* xs;
    SimpleCrossSection* temp;
    bool cudaCopyMade;

public:
    SimpleCrossSection* xs_device;

};

gpuFloatType_t launchGetTotalXS( SimpleCrossSectionHost* pXS, gpuFloatType_t energy);


}


#endif /* SIMPLECROSSSECTION_HH_ */
