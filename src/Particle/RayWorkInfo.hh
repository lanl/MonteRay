#ifndef RAYWORKINFO_HH_
#define RAYWORKINFO_HH_

#include <string>

#include "MonteRayTypes.hh"
#include "MonteRayCopyMemory.hh"

namespace MonteRay{

template<unsigned N = 1>
class RayWorkInfo : public CopyMemoryBase<RayWorkInfo<N>>{
public:
    using Base = CopyMemoryBase<RayWorkInfo<N>>;

    /// Primary RayWorkInfo constructor.
    /// Takes the size of the list as an argument.
    CUDAHOST_CALLABLE_MEMBER RayWorkInfo(unsigned num, bool cpuAllocate = false );

    CUDAHOST_CALLABLE_MEMBER ~RayWorkInfo();

    CUDAHOST_CALLABLE_MEMBER std::string className() { return std::string("RayWorkInfo");}

    CUDAHOST_CALLABLE_MEMBER void reallocate(unsigned n);

    CUDAHOST_CALLABLE_MEMBER void init() {
        nAllocated = 0;
        indices = NULL;
        rayCastSize = NULL;
        rayCastCell = NULL;
        rayCastDistance = NULL;
        crossingSize = NULL;
        crossingCell = NULL;
        crossingDistance = NULL;
    }

    CUDA_CALLABLE_MEMBER void clear(void) {
        if( allocateOnCPU ) {
            for( unsigned i = 0; i< nAllocated; ++i ) {
                rayCastSize[i] = 0;
            }
            for( unsigned i = 0; i< nAllocated*3; ++i ) {
                crossingSize[i] = 0;
            }
        }
    }

    CUDAHOST_CALLABLE_MEMBER void copy(const RayWorkInfo<N>* rhs);

    CUDA_CALLABLE_MEMBER unsigned capacity(void) const {
        return nAllocated;
    }

    CUDA_CALLABLE_MEMBER unsigned getIndex(unsigned dim, unsigned i) const {
        return indices[i+nAllocated*dim];
    }

    CUDA_CALLABLE_MEMBER void setIndex(unsigned dim, unsigned i, int index) {
        indices[i+nAllocated*dim] = index;
    }

    CUDA_CALLABLE_MEMBER void clear( unsigned i) {
        getRayCastSize(i) = 0;
        getCrossingSize(0, i) = 0;
        getCrossingSize(1, i) = 0;
        getCrossingSize(2, i) = 0;
    }

    CUDA_CALLABLE_MEMBER int& getRayCastSize( unsigned i) const {
        return rayCastSize[i];
    }

    CUDA_CALLABLE_MEMBER void addRayCastCell(unsigned i, int cellID, gpuRayFloat_t dist);

    CUDA_CALLABLE_MEMBER int& getRayCastCell(unsigned i, unsigned cell) const {
        return rayCastCell[i*MAXNUMRAYCELLS + cell];
    }

    CUDA_CALLABLE_MEMBER gpuRayFloat_t& getRayCastDist(unsigned i, unsigned cell) const {
        return rayCastDistance[i*MAXNUMRAYCELLS + cell];
    }

    CUDA_CALLABLE_MEMBER int& getCrossingSize(unsigned dim, unsigned i) const {
        return crossingSize[dim*nAllocated + i];
    }

    CUDA_CALLABLE_MEMBER void addCrossingCell(unsigned dim, unsigned i, int cellID, gpuRayFloat_t dist);

    CUDA_CALLABLE_MEMBER int& getCrossingCell(unsigned dim, unsigned i, unsigned cell) const {
        return crossingCell[dim*nAllocated*MAXNUMVERTICES + i*MAXNUMVERTICES + cell];
    }

    CUDA_CALLABLE_MEMBER gpuRayFloat_t& getCrossingDist(unsigned dim, unsigned i, unsigned cell) const {
        return crossingDistance[dim*nAllocated*MAXNUMVERTICES + i*MAXNUMVERTICES + cell];
    }

// Data
    bool allocateOnCPU = false;
    unsigned nAllocated = 0;

    int* indices = NULL;

    int* rayCastSize = NULL;
    int* rayCastCell = NULL;
    gpuRayFloat_t* rayCastDistance = NULL;

    int* crossingSize = NULL;
    int* crossingCell = NULL;
    gpuRayFloat_t* crossingDistance = NULL;
};

} // end namespace

#endif /* RAYWORKINFO_HH_ */
