#ifndef MONTERAYSPHERICALGRID_HH_
#define MONTERAYSPHERICALGRID_HH_

#include "MonteRayTypes.hh"
#include "MonteRay_GridSystemInterface.hh"
#include "RayWorkInfo.hh"

namespace MonteRay {

class MonteRay_SphericalGrid {
public:
    typedef MonteRay_GridBins::Position_t Position_t;
    typedef MonteRay_GridBins::Direction_t Direction_t;

    enum coord {R=0,DimMax=1};
    int DIM = 1;

    using GridBins_t = MonteRay_GridBins;
    using pGridInfo_t = GridBins_t*;
    using pArrayOfpGridInfo_t = pGridInfo_t[3];

    MonteRay_SphericalGrid(unsigned d, pArrayOfpGridInfo_t pBins);
    MonteRay_SphericalGrid(unsigned d, GridBins_t* );

    CUDA_CALLABLE_MEMBER auto dimension() const { return DIM; }

    CUDA_CALLABLE_MEMBER void validate(void);
    CUDA_CALLABLE_MEMBER void validateR(void);

    CUDA_CALLABLE_MEMBER unsigned getNumRBins() const { return numRBins; }
    CUDA_CALLABLE_MEMBER unsigned getNumBins( unsigned d) const { return d == 0 ? numRBins : 0; }

    CUDA_CALLABLE_MEMBER gpuRayFloat_t getRVertex(unsigned i) const { return pRVertices->vertices[i]; }
    CUDA_CALLABLE_MEMBER gpuRayFloat_t getRSqVertex(unsigned i) const { return pRVertices->verticesSq[i]; }

    CUDA_CALLABLE_MEMBER Position_t convertFromCartesian( const Position_t& pos) const;

    CUDA_CALLABLE_MEMBER int getRadialIndexFromR( gpuRayFloat_t R ) const { return pRVertices->getRadialIndexFromR(R); }
    CUDA_CALLABLE_MEMBER int getRadialIndexFromRSq( gpuRayFloat_t RSq ) const { return pRVertices->getRadialIndexFromRSq(RSq); }

    CUDA_CALLABLE_MEMBER unsigned getIndex( const GridBins_t::Position_t& particle_pos) const;
    CUDA_CALLABLE_MEMBER bool isIndexOutside( unsigned d,  int i) const;

    CUDA_CALLABLE_MEMBER bool isOutside(  const int i[]) const;

    CUDA_CALLABLE_MEMBER unsigned calcIndex( const int[] ) const;

    CUDA_CALLABLE_MEMBER uint3 calcIJK( unsigned index ) const { return {index, 0, 0}; }

    CUDA_CALLABLE_MEMBER gpuRayFloat_t getVolume( unsigned index ) const;

    CUDA_CALLABLE_MEMBER
    void rayTrace( const unsigned threadID,
              RayWorkInfo& rayInfo,
              const GridBins_t::Position_t& pos,
              const GridBins_t::Position_t& dir,
              const gpuRayFloat_t distance,
              const bool outsideDistances=false) const;

    CUDA_CALLABLE_MEMBER
    void crossingDistance(
            const unsigned dim,
            const unsigned threadID,
            RayWorkInfo& rayInfo,
            const GridBins_t::Position_t& pos,
            const GridBins_t::Position_t& dir,
            const gpuRayFloat_t distance ) const;

protected:

    CUDA_CALLABLE_MEMBER
    void
    radialCrossingDistances(
            const unsigned dim,
            const unsigned threadID,
            RayWorkInfo& rayInfo,
            const Position_t& pos,
            const Direction_t& dir,
            unsigned rIndex,
            const gpuRayFloat_t distance ) const;

    CUDA_CALLABLE_MEMBER
    void
    radialCrossingDistances(
            const unsigned dim,
            const unsigned threadID,
            RayWorkInfo& rayInfo,
            const Position_t& pos,
            const Direction_t& dir,
            const gpuRayFloat_t distance ) const;

    template<bool OUTWARD>
    CUDA_CALLABLE_MEMBER
    void
    radialCrossingDistancesSingleDirection(
            const unsigned dim,
            const unsigned threadID,
            RayWorkInfo& rayInfo,
            const Position_t& pos,
            const Direction_t& dir,
            const gpuRayFloat_t distance ) const;

protected:
    CUDA_CALLABLE_MEMBER gpuRayFloat_t calcParticleRSq( const gpuRayFloat_t&  pos) const { return pos*pos; }
    CUDA_CALLABLE_MEMBER gpuRayFloat_t calcParticleRSq( const Position_t&  pos) const { return pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]; }
    CUDA_CALLABLE_MEMBER gpuRayFloat_t calcQuadraticA(  const Direction_t& dir) const { return 1.0; } //=dot(pDirection,pDirection)
    CUDA_CALLABLE_MEMBER gpuRayFloat_t calcQuadraticB(  const Position_t&  pos, const Direction_t& dir) const { return 2.0*(pos[0]*dir[0] + pos[1]*dir[1] + pos[2]*dir[2]); }

public:
    MonteRay_SphericalGrid** ptrDevicePtr = nullptr;
    MonteRay_SphericalGrid* devicePtr = nullptr;

private:
    pGridInfo_t pRVertices = nullptr;
    unsigned numRBins = 0;
    //bool regular = false;

    const bool debug = false;
};

} /* namespace MonteRay */

#endif /* MONTERAYSPHERICALGRID_HH_ */
