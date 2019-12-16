#ifndef MONTERAYCARTESIANGRID_HH_
#define MONTERAYCARTESIANGRID_HH_

#include "MonteRayTypes.hh"
#include "RayWorkInfo.hh"
#include "MonteRay_GridSystemInterface.hh"
#include "ThirdParty/Array.hh"

namespace MonteRay {

class MonteRay_CartesianGrid : public MonteRay_GridSystemInterface {
public:
  enum coord {X,Y,Z,DimMax};

public:
    MonteRay_CartesianGrid() = default;

    MonteRay_CartesianGrid(int d, GridBins_t, GridBins_t, GridBins_t );
    template <typename GridBinsArray>
    MonteRay_CartesianGrid(int d, GridBinsArray bins): MonteRay_CartesianGrid(d, std::move(bins[0]), std::move(bins[1]), std::move(bins[2])) { }

    CUDA_CALLABLE_MEMBER auto dimension() const { return DIM; }

    CUDA_CALLABLE_MEMBER unsigned getIndex(const GridBins_t::Position_t& particle_pos) const;

    CUDA_CALLABLE_MEMBER int getDimIndex( unsigned d, gpuRayFloat_t pos) const {return gridBins[d].getLinearIndex( pos ); }

    CUDA_CALLABLE_MEMBER gpuRayFloat_t getVolume( unsigned index ) const;

    CUDA_CALLABLE_MEMBER uint3 calcIJK( unsigned index ) const;

    CUDA_CALLABLE_MEMBER unsigned getNumBins( unsigned d) const;

    CUDA_CALLABLE_MEMBER
    DirectionAndSpeed convertToCellReferenceFrame(
      const Vector3D<gpuRayFloat_t>& cellVelocity,
      const GridBins_t::Position_t&, // unused, exists to maintain same API as cylindrical and spherical grid
      GridBins_t::Direction_t dir,
      gpuRayFloat_t speed) const;
    
    CUDA_CALLABLE_MEMBER 
    auto calcIndices(const GridBins_t::Position_t& pos) const {
      return Array<int, 3>{ getDimIndex(0, pos[0] ),
                            getDimIndex(1, pos[1] ),
                            getDimIndex(2, pos[2] ) };
    }
    
    CUDA_CALLABLE_MEMBER 
    DistAndDir getMinDistToSurface( 
        const GridBins_t::Position_t& pos, 
        const GridBins_t::Direction_t& dir, 
        const int indices[]) const;

    CUDA_CALLABLE_MEMBER
    gpuRayFloat_t getDistanceToInsideOfMesh(const GridBins_t::Position_t& pos, const GridBins_t::Direction_t& dir) const;

    CUDA_CALLABLE_MEMBER
    void rayTrace( const unsigned threadID,
              RayWorkInfo& rayInfo,
              const GridBins_t::Position_t& particle_pos,
              const GridBins_t::Position_t& particle_dir,
              const gpuRayFloat_t distance,
              const bool outsideDistances=false ) const ;

    CUDA_CALLABLE_MEMBER
    void crossingDistance(  const unsigned dim,
                       const unsigned threadID,
                       RayWorkInfo& rayInfo,
                       const GridBins_t::Position_t& pos,
                       const GridBins_t::Direction_t& dir,
                       const gpuRayFloat_t distance ) const;


    CUDA_CALLABLE_MEMBER
    void crossingDistance(  const unsigned dim,
                       const unsigned threadID,
                       RayWorkInfo& rayInfo,
                       const gpuRayFloat_t pos,
                       const gpuRayFloat_t dir,
                       const gpuRayFloat_t distance ) const;
private:

    CUDA_CALLABLE_MEMBER
    void crossingDistance( const unsigned dim,
                      const unsigned threadID,
                      RayWorkInfo& rayInfo,
                      const GridBins_t& Bins,
                      const gpuRayFloat_t pos,
                      const gpuRayFloat_t dir,
                      const gpuRayFloat_t distance,
                      const bool equal_spacing=false) const;

};

} /* namespace MonteRay */

#endif /* MONTERAYCARTESIANGRID_HH_ */
