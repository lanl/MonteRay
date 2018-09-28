#ifndef HASHBINS_H_
#define HASHBINS_H_

#include "MonteRayTypes.hh"
#include "MonteRayCopyMemory.hh"

namespace MonteRay{

class HashBins : public CopyMemoryBase<HashBins> {
private:
    gpuFloatType_t min = 0.0;
    gpuFloatType_t max = 0.0 ;
    gpuFloatType_t invDelta = 0.0;
    unsigned nEdges = 0;
    unsigned* binBounds = nullptr;

public:
    using Base = MonteRay::CopyMemoryBase<HashBins>;

    HashBins(gpuFloatType_t *vertices, unsigned nVertices, unsigned nHashBinEdges = 8000);

    ~HashBins();

    std::string className(){ return std::string("HashBins");}

    void init();

    void copy(const HashBins* rhs);

    CUDA_CALLABLE_MEMBER
    gpuFloatType_t getMin(void) const { return min; }

    CUDA_CALLABLE_MEMBER
    gpuFloatType_t getMax(void) const { return max; }

    CUDA_CALLABLE_MEMBER
    gpuFloatType_t getDelta(void) const { return 1/invDelta; }

    CUDA_CALLABLE_MEMBER
    gpuFloatType_t getBinEdge(unsigned i) const { return (1/invDelta)*i+min; }

    CUDA_CALLABLE_MEMBER
    unsigned getBinBound( unsigned i ) const { return binBounds[i]; }

    CUDA_CALLABLE_MEMBER
    unsigned getNEdges( void ) const { return nEdges; }

    CUDA_CALLABLE_MEMBER
    void getLowerUpperBins( double value, unsigned& lower, unsigned& upper) const;

};


}
#endif /* HASHBINS_H_ */
