#ifndef SIMPLECROSSSECTION_HH_
#define SIMPLECROSSSECTION_HH_

#include <iostream>

#include "gpuGlobal.h"

struct SimpleCrossSection {
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
unsigned getIndex(struct SimpleCrossSection* pXS, gpuFloatType_t E );

#ifdef CUDA
__device__ __host__
#endif
gpuFloatType_t getAWR(struct SimpleCrossSection* pXS);

#ifdef CUDA
__global__ void kernelGetTotalXS(struct SimpleCrossSection* pXS, gpuFloatType_t E, gpuFloatType_t* result);
#endif

class ContinuousNeutron;

class SimpleCrossSectionHost {
public:
    SimpleCrossSectionHost(unsigned num);
    ~SimpleCrossSectionHost();

    void copyToGPU(void);

    unsigned size(void) const { return xs->numPoints; }
    gpuFloatType_t getEnergy(unsigned i) const { return ::getEnergy(xs, i); }
    gpuFloatType_t getTotalXSByIndex(unsigned i) const { return ::getTotalXSByIndex(xs, i); }

    gpuFloatType_t getTotalXS( gpuFloatType_t E ) const { return ::getTotalXS(xs, E); }

    void setEnergy(unsigned i, gpuFloatType_t e) { xs->energies[i] = e; }
    unsigned getIndex( gpuFloatType_t e ) const { return ::getIndex( xs, e); }

    void setTotalXS(unsigned i, gpuFloatType_t value) { xs->totalXS[i] = value; }

    void setTotalXS(unsigned i, gpuFloatType_t E, gpuFloatType_t value) {
        xs->energies[i] = E;
        xs->totalXS[i] = value;
    }

    gpuFloatType_t getAWR(void) const {return ::getAWR(xs); }
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

private:
    struct SimpleCrossSection* xs;
    SimpleCrossSection* temp;
    bool cudaCopyMade;

public:
    SimpleCrossSection* xs_device;

};


#endif /* SIMPLECROSSSECTION_HH_ */
