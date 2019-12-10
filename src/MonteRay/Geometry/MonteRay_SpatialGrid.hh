#ifndef MONTERAY_SPATIALGRID_HH_
#define MONTERAY_SPATIALGRID_HH_

#include "MonteRay_TransportMeshTypeEnum.hh"

#include <memory>
#include "MonteRayVector3D.hh"

#include "RayWorkInfo.hh"
#include "MaterialProperties.hh"

#include "ManagedAllocator.hh"
#include "SimpleVector.hh"
#include "ThirdParty/Array.hh"

#include "MonteRay_CylindricalGrid.t.hh"
#include "MonteRay_CartesianGrid.t.hh"

#include "MonteRay_GridVariant.hh"

namespace MonteRay {

class MonteRay_SpatialGrid : public Managed {
public:
  using GridGeometry_t = GridVariant;
  using GridBins_t = MonteRay_GridBins;
  using Index_t = unsigned;
private:
  GridGeometry_t gridVariant;

public:
  enum indexCartEnum_t {CART_X=0, CART_Y=1, CART_Z=2};
  enum indexCylEnum_t  {CYLR_R=0, CYLR_Z=1, CYLR_THETA=2};
  enum indexSphEnum_t  {SPH_R=0};
  enum { MaxDim=3 };

  using Position_t = Vector3D<gpuRayFloat_t>;
  using Direction_t = Vector3D<gpuRayFloat_t>;

  //TRA/JES Move to GridBins -- ?
  static const unsigned OUTSIDE_MESH = UINT_MAX;

  template<typename Reader_t>
  CUDAHOST_CALLABLE_MEMBER
  MonteRay_SpatialGrid( Reader_t& reader ) {
    int DIM = 1;
    TransportMeshType coordSystem;
    GridBins_t::coordinate_t binLayout[3] = {GridBins_t::coordinate_t::LINEAR, GridBins_t::coordinate_t::LINEAR, GridBins_t::coordinate_t::LINEAR};
    if( reader.getGeometryString() == "XYZ" )  {
      coordSystem = TransportMeshType::Cartesian;
      DIM = 3;
    } else if ( reader.getGeometryString() == "RZ" )  {
      coordSystem = TransportMeshType::Cylindrical;
      binLayout[0] = GridBins_t::coordinate_t::RADIAL;
      DIM = 2;
    } else {
      throw std::runtime_error( "MonteRay_SpatialGrid(reader) -- Geometry type " + reader.getGeometryString() + " not yet supported." );
    }

    MonteRay_GridSystemInterface::GridBinsArray_t gridBins;
    for( int d=0; d < DIM; ++d) {
      gridBins[d] = GridBins_t{reader.getVertices(d), binLayout[d]};
    }

    gridVariant = GridGeometry_t(coordSystem, std::move(gridBins));
  }


  template<typename GridBins>
  MonteRay_SpatialGrid(TransportMeshType meshType, GridBins&& gridBins) {
    unsigned long long nBins = 1;
    for (const auto& gridBin : gridBins){
      if( gridBin.numBins() == 0 ) {
        std::stringstream msg;
        msg << "No bins found in GridBins used to construct MonteRay_SpatialGrid! \n "
            << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_SpatialGrid constructor \n\n";
        throw std::runtime_error( msg.str() );
      }
      nBins *= gridBin.getNumBins();
    }
    if (nBins > UINT_MAX){
      throw std::runtime_error("Number of bins created while constructing MonteRay_SpatialGrid exceeds maximum value of " + std::to_string(UINT_MAX));
    }
    //TODO: TRA - need 1d,2d for Cart_regular and Cart?
    gridVariant = GridGeometry_t(meshType, gridBins);
  }

  /* template <typename... Args> */
  /* MonteRay_SpatialGrid(Args&&... args) : gridVariant(std::forward<Args>(args)...) {} */

  CUDAHOST_CALLABLE_MEMBER std::string className(){ return std::string("MonteRay_SpatialGrid");}

  CUDA_CALLABLE_MEMBER
  TransportMeshType getCoordinateSystem() const {
    return gridVariant.getCoordinateSystem();
  }

  CUDA_CALLABLE_MEMBER
  unsigned getDimension() const { 
    return gridVariant.visit( [](const auto& grid){ return grid.dimension(); });
  }
  CUDA_CALLABLE_MEMBER
  unsigned dimension() const { return getDimension(); }

  CUDA_CALLABLE_MEMBER
  unsigned getNumGridBins(unsigned index) const { 
    return gridVariant.visit( [=](const auto& grid){ return grid.getNumBins(index); } );
  }

  CUDA_CALLABLE_MEMBER
  gpuRayFloat_t getMinVertex(unsigned index) const { 
    return gridVariant.visit( [=](const auto& grid){ return grid.getMinVertex(index); });
  }

  CUDA_CALLABLE_MEMBER
  gpuRayFloat_t getMaxVertex(unsigned index) const { 
    return gridVariant.visit( [=](const auto& grid){ return grid.getMaxVertex(index); });
  }

  CUDA_CALLABLE_MEMBER
  gpuRayFloat_t getVertex(unsigned d, unsigned index) const { 
    return gridVariant.visit( [=](const auto& grid){ return grid.getVertex(d, index); });
  }

  CUDA_CALLABLE_MEMBER
  unsigned getIndex(Position_t pos) const { 
    return gridVariant.visit( [&](const auto& grid){ return grid.getIndex(pos); } );
  }

  CUDA_CALLABLE_MEMBER
  gpuRayFloat_t getVolume(unsigned index) const { 
    return gridVariant.visit( [=](const auto& grid){ return grid.getVolume(index); } );
  }

  CUDA_CALLABLE_MEMBER
  unsigned getNumCells(void) const {
    unsigned nCells = 1;
    for( int d=0; d < dimension(); ++d  ){
      nCells *= this->getNumGridBins(d);
    }
    return nCells;
  }

  CUDA_CALLABLE_MEMBER
  size_t numCells(void) const {return getNumCells();}

  CUDA_CALLABLE_MEMBER
  size_t size(void) const {return getNumCells();}

  CUDA_CALLABLE_MEMBER
  size_t getNumVertices(unsigned i) const { 
    return gridVariant.visit( [=](const auto& grid){ return grid.getNumVertices(i); });
  }

  CUDA_CALLABLE_MEMBER
  size_t getNumVerticesSq(unsigned i) const { 
    return gridVariant.visit( [=](const auto& grid){ return grid.getNumVerticesSq(i); });
  };

  template<class Particle>
  CUDA_CALLABLE_MEMBER
  unsigned getIndex(const Particle& p) const {
      Position_t particle_pos = p.getPosition();
      return this->getIndex( particle_pos );
  }

  CUDA_CALLABLE_MEMBER
  gpuRayFloat_t returnCellVolume( unsigned index ) const { return this->getVolume( index ); }

  CUDA_CALLABLE_MEMBER
  unsigned rayTrace( const unsigned threadID,
            RayWorkInfo& rayInfo,
            const Position_t& pos,
            const Position_t& dir,
            const gpuRayFloat_t distance,
            const bool outsideDistances=false) const {
    gridVariant.visit( [&](const auto& grid) { grid.rayTrace(threadID, rayInfo, pos, dir, distance, outsideDistances); } );
    return rayInfo.getRayCastSize(threadID);
  }

  template<class Ray>
  CUDA_CALLABLE_MEMBER
  auto rayTrace(  const unsigned threadID,
                  RayWorkInfo& rayInfo,
                  const Ray& ray,
                  const gpuRayFloat_t distance,
                  const bool OutsideDistances=false ) const {
    return rayTrace( threadID, rayInfo, ray.getPosition(), ray.getDirection(), distance, OutsideDistances );
  }

  CUDA_CALLABLE_MEMBER
  void crossingDistance( const unsigned dim,
                    const unsigned threadID,
                    RayWorkInfo& rayInfo,
                    const gpuRayFloat_t pos,
                    const gpuRayFloat_t dir,
                    const gpuRayFloat_t distance) const {
    Position_t position; position[dim] = pos;
    Direction_t direction; direction[dim] = dir;
    gridVariant.visit( [&](const auto& grid){ crossingDistance(dim, threadID, rayInfo, position, direction, distance ); });
  }

  CUDA_CALLABLE_MEMBER
  void crossingDistance( const unsigned dim,
                    const unsigned threadID,
                    RayWorkInfo& rayInfo,
                    const Position_t& pos,
                    const Direction_t& dir,
                    const gpuRayFloat_t distance) const {
    gridVariant.visit( [&](const auto& grid){ grid.crossingDistance(dim, threadID, rayInfo, pos, dir, distance ); });
  }

private:
  void checkDim( unsigned dim ) const;

public:
  void write(std::ostream& outfile) const;

  /* static MonteRay_SpatialGrid read( std::istream& infile ); */

};

} /* namespace MonteRay */

#endif /* MONTERAY_SPATIALGRID_HH_ */
