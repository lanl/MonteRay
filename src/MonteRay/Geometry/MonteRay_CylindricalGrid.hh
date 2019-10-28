#ifndef MONTERAYCYLINDRICALGRID_HH_
#define MONTERAYCYLINDRICALGRID_HH_

#include "MonteRayTypes.hh"
#include "MonteRay_GridSystemInterface.hh"
#include "RayWorkInfo.hh"

namespace MonteRay {

class MonteRay_CylindricalGrid : public MonteRay_GridSystemInterface {
public:
    typedef MonteRay_GridBins::Position_t Position_t;
    typedef MonteRay_GridBins::Direction_t Direction_t;

    enum coord {R=0,Z=1,Theta=2,DimMax=2};  //Theta not supported
    enum cart_coord {x=0, y=1, z=2};

    using GridBins_t = MonteRay_GridBins;
    using pGridInfo_t = GridBins_t*;
    using pArrayOfpGridInfo_t = pGridInfo_t[3];

    CUDA_CALLABLE_MEMBER MonteRay_CylindricalGrid(unsigned d, pArrayOfpGridInfo_t pBins);
    CUDA_CALLABLE_MEMBER MonteRay_CylindricalGrid(unsigned d, GridBins_t* pGridR, GridBins_t* pGridZ );

    CUDA_CALLABLE_MEMBER virtual ~MonteRay_CylindricalGrid(void);

    CUDAHOST_CALLABLE_MEMBER void copyToGPU(void);

    CUDAHOST_CALLABLE_MEMBER
    MonteRay_CylindricalGrid* getDeviceInstancePtr();

    CUDA_CALLABLE_MEMBER void validate(void);
    CUDA_CALLABLE_MEMBER void validateR(void);

    CUDA_CALLABLE_MEMBER unsigned getNumBins( unsigned d) const;
    CUDA_CALLABLE_MEMBER unsigned getNumRBins() const { return numRBins; }
    CUDA_CALLABLE_MEMBER unsigned getNumZBins() const { return numZBins; }
    //CUDA_CALLABLE_MEMBER unsigned getNumThetaBins() const { return numThetaBins; }

    CUDA_CALLABLE_MEMBER gpuRayFloat_t getRVertex(unsigned i) const { return pRVertices->vertices[i]; }
    CUDA_CALLABLE_MEMBER gpuRayFloat_t getRSqVertex(unsigned i) const { return pRVertices->verticesSq[i]; }
    CUDA_CALLABLE_MEMBER gpuRayFloat_t getZVertex(unsigned i) const { return pZVertices->vertices[i]; }
    //CUDA_CALLABLE_MEMBER gpuRayFloat_t getThetaVertex(unsigned i) const { return pThetaVertices->vertices.at(i); }

    CUDA_CALLABLE_MEMBER Position_t convertFromCartesian( const Position_t& pos) const;

    CUDA_CALLABLE_MEMBER int getRadialIndexFromR( gpuRayFloat_t R ) const { return pRVertices->getRadialIndexFromR(R); }
    CUDA_CALLABLE_MEMBER int getRadialIndexFromRSq( gpuRayFloat_t RSq ) const { return pRVertices->getRadialIndexFromRSq(RSq); }
    CUDA_CALLABLE_MEMBER int getAxialIndex( gpuRayFloat_t z) const { return pZVertices->getLinearIndex(z);}

    CUDA_CALLABLE_MEMBER unsigned getIndex( const GridBins_t::Position_t& particle_pos) const;
    CUDA_CALLABLE_MEMBER bool isIndexOutside( unsigned d,  int i) const;

    CUDA_CALLABLE_MEMBER bool isOutside(  const int i[]) const;

    CUDA_CALLABLE_MEMBER unsigned calcIndex( const int[] ) const;

    CUDA_CALLABLE_MEMBER uint3 calcIJK( unsigned index ) const;

    CUDA_CALLABLE_MEMBER gpuRayFloat_t getVolume( unsigned index ) const;

    CUDA_CALLABLE_MEMBER
    void
    rayTrace(
            const unsigned threadID,
            RayWorkInfo& rayInfo,
            const GridBins_t::Position_t& particle_pos,
            const GridBins_t::Position_t& particle_dir,
            const gpuRayFloat_t distance,
            const bool outsideDistances=false ) const;

    CUDA_CALLABLE_MEMBER
    virtual void
    rayTraceWithMovingMaterials( const unsigned threadID,
              RayWorkInfo& rayInfo,
              const GridBins_t::Position_t& particle_pos,
              const GridBins_t::Position_t& particle_dir,
              const gpuRayFloat_t distance,
              const gpuRayFloat_t speed,
              const MaterialProperties& matProps,
              const bool outsideDistances=false ) const {}

    CUDA_CALLABLE_MEMBER
    void
    crossingDistance(
            const unsigned dim,
            const unsigned threadID,
            RayWorkInfo& rayInfo,
            const GridBins_t::Position_t& pos,
            const GridBins_t::Direction_t& dir,
            const gpuRayFloat_t distance ) const;

    CUDA_CALLABLE_MEMBER
    void
    radialCrossingDistances(
            const unsigned dim,
            const unsigned threadID,
            RayWorkInfo& rayInfo,
            const Position_t& pos,
            const Direction_t& dir,
            const double rSq,
            const unsigned rIndex,
            const gpuRayFloat_t distance) const;

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
            unsigned dim,
            unsigned threadID,
            RayWorkInfo& rayInfo,
            const Position_t& pos,
            const Direction_t& dir,
            const gpuRayFloat_t distance ) const;

protected:
    CUDA_CALLABLE_MEMBER gpuRayFloat_t calcParticleRSq( const Position_t&  pos) const { return pos[x]*pos[x] + pos[y]*pos[y]; }
    CUDA_CALLABLE_MEMBER gpuRayFloat_t calcQuadraticA(  const Direction_t& dir) const { return dir[x]*dir[x] + dir[y]*dir[y]; } //=dot(pDirection,pDirection)
    CUDA_CALLABLE_MEMBER gpuRayFloat_t calcQuadraticB(  const Position_t&  pos, const Direction_t& dir) const { return 2.0*(pos[x]*dir[x] + pos[y]*dir[y]); }

public:
    MonteRay_CylindricalGrid** ptrDevicePtr = nullptr;
    MonteRay_CylindricalGrid* devicePtr = nullptr;

private:
    static constexpr gpuRayFloat_t inf = std::numeric_limits<gpuRayFloat_t>::infinity();
    static const unsigned COORD_DIM = 2;

    pGridInfo_t pRVertices = nullptr;
    pGridBins_t pZVertices = nullptr;
    // pGridBins_t pThetaVertices = nullptr;

    unsigned numRBins = 0;
    unsigned numZBins = 0;
    //unsigned numThetaBins = 0;

    //bool regular = false;

#ifndef NDEBUG
    static const bool debug = false;
#endif
};

} /* namespace MonteRay */

#endif /* MONTERAYCYLINDRICALGRID_HH_ */
