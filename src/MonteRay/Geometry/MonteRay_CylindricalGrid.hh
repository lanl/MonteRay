#ifndef MONTERAYCYLINDRICALGRID_HH_
#define MONTERAYCYLINDRICALGRID_HH_

#include "MonteRayTypes.hh"
#include "MonteRay_GridSystemInterface.hh"
#include "RayWorkInfo.hh"

namespace MonteRay {

class MonteRay_CylindricalGrid : public MonteRay_GridSystemInterface {
public:
    using GridBins_t = MonteRay_GridBins;
    using Position_t = GridBins_t::Position_t;
    using Direction_t = GridBins_t::Direction_t;

    enum coord {R=0,CZ=1,Theta=2,DimMax=2};  //Theta not supported
    enum cart_coord {x=0, y=1, z=2};
private:

    static constexpr gpuRayFloat_t inf = std::numeric_limits<gpuRayFloat_t>::infinity();
    static const int COORD_DIM = 2; // TODO: not static const?
public:

    MonteRay_CylindricalGrid() = default;
    MonteRay_CylindricalGrid(int d, GridBins_t gridR, GridBins_t gridZ );
    template <typename GridBins>
    MonteRay_CylindricalGrid(int d, GridBins gridBins) : MonteRay_CylindricalGrid(d, std::move(gridBins[0]), std::move(gridBins[1])){ }

    CUDA_CALLABLE_MEMBER unsigned getNumRBins() const { return gridBins[R].getNumBins(); }
    CUDA_CALLABLE_MEMBER unsigned getNumZBins() const { return gridBins[CZ].getNumBins(); }
    CUDA_CALLABLE_MEMBER unsigned getNumBins(int d) const;

    CUDA_CALLABLE_MEMBER gpuRayFloat_t getRVertex(unsigned i) const { return gridBins[R].vertices[i]; }
    CUDA_CALLABLE_MEMBER gpuRayFloat_t getRSqVertex(unsigned i) const { return gridBins[R].verticesSq[i]; }
    CUDA_CALLABLE_MEMBER gpuRayFloat_t getZVertex(unsigned i) const { return gridBins[CZ].vertices[i]; }

    CUDA_CALLABLE_MEMBER Position_t convertFromCartesian( const Position_t& pos) const;

    CUDA_CALLABLE_MEMBER int getRadialIndexFromR( gpuRayFloat_t R ) const { return gridBins[R].getRadialIndexFromR(R); }
    CUDA_CALLABLE_MEMBER int getRadialIndexFromRSq( gpuRayFloat_t RSq ) const { return gridBins[R].getRadialIndexFromRSq(RSq); }
    CUDA_CALLABLE_MEMBER int getAxialIndex( gpuRayFloat_t z) const { return gridBins[CZ].getLinearIndex(z);}

    CUDA_CALLABLE_MEMBER unsigned getIndex( const GridBins_t::Position_t& particle_pos) const;
    CUDA_CALLABLE_MEMBER bool isIndexOutside( unsigned d,  int i) const;

    CUDA_CALLABLE_MEMBER auto dimension() const { return DIM; }

    CUDA_CALLABLE_MEMBER bool isOutside(  const int i[]) const;

    CUDA_CALLABLE_MEMBER uint3 calcIJK( unsigned index ) const;

    CUDA_CALLABLE_MEMBER gpuRayFloat_t getVolume( unsigned index ) const;

    CUDA_CALLABLE_MEMBER
    DirectionAndSpeed convertToCellReferenceFrame(
        const Vector3D<gpuRayFloat_t>& cellVelocity,
        const GridBins_t::Position_t& pos,
        GridBins_t::Direction_t dir,
        gpuRayFloat_t speed) const;
    
    CUDA_CALLABLE_MEMBER 
    Array<int, 3> calcIndices(const GridBins_t::Position_t& pos) const;

    CUDA_CALLABLE_MEMBER 
    DistAndDir getMinRadialDistAndDir( 
        const GridBins_t::Position_t& pos, 
        const GridBins_t::Direction_t& dir, 
        const int radialIndex) const;
    
    CUDA_CALLABLE_MEMBER 
    DistAndDir getMinDistToSurface( 
        const GridBins_t::Position_t& pos, 
        const GridBins_t::Direction_t& dir, 
        const int indices[]) const;

    CUDA_CALLABLE_MEMBER 
    constexpr bool isMovingInward(
            const GridBins_t::Position_t& pos,
            const GridBins_t::Position_t& dir) const {
      // unnormalized 'normal' * dir for circle drawn through pos w/ origin (0,0)
      return Math::signbit(pos[x]*dir[x] + pos[y]*dir[y]); 
    }

    CUDA_CALLABLE_MEMBER
    gpuRayFloat_t getDistanceToInsideOfMesh(const GridBins_t::Position_t& pos, const GridBins_t::Direction_t& dir) const;

    CUDA_CALLABLE_MEMBER
    void rayTrace(
            const unsigned threadID,
            RayWorkInfo& rayInfo,
            const GridBins_t::Position_t& particle_pos,
            const GridBins_t::Position_t& particle_dir,
            const gpuRayFloat_t distance,
            const bool outsideDistances=false ) const;

    CUDA_CALLABLE_MEMBER
    void crossingDistance(
            const unsigned dim,
            const unsigned threadID,
            RayWorkInfo& rayInfo,
            const GridBins_t::Position_t& pos,
            const GridBins_t::Direction_t& dir,
            const gpuRayFloat_t distance ) const;

    CUDA_CALLABLE_MEMBER
    void radialCrossingDistances(
            const unsigned dim,
            const unsigned threadID,
            RayWorkInfo& rayInfo,
            const Position_t& pos,
            const Direction_t& dir,
            const double rSq,
            const unsigned rIndex,
            const gpuRayFloat_t distance) const;

    CUDA_CALLABLE_MEMBER
    void radialCrossingDistances(
            const unsigned dim,
            const unsigned threadID,
            RayWorkInfo& rayInfo,
            const Position_t& pos,
            const Direction_t& dir,
            const gpuRayFloat_t distance ) const;

    template<bool OUTWARD>
    CUDA_CALLABLE_MEMBER
    void radialCrossingDistancesSingleDirection(
            unsigned dim,
            unsigned threadID,
            RayWorkInfo& rayInfo,
            const Position_t& pos,
            const Direction_t& dir,
            const gpuRayFloat_t distance ) const;

protected:
    CUDA_CALLABLE_MEMBER gpuRayFloat_t calcParticleRSq( const Position_t&  pos) const { return pos[x]*pos[x] + pos[y]*pos[y]; }
    CUDA_CALLABLE_MEMBER gpuRayFloat_t calcQuadraticA(  const Direction_t& dir) const { return dir[x]*dir[x] + dir[y]*dir[y]; } 
    CUDA_CALLABLE_MEMBER gpuRayFloat_t calcQuadraticB(  const Position_t&  pos, const Direction_t& dir) const { return 2.0*(pos[x]*dir[x] + pos[y]*dir[y]); }
};

} /* namespace MonteRay */

#endif /* MONTERAYCYLINDRICALGRID_HH_ */
