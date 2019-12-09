#ifndef MonteRay_GridBins_HH_
#define MonteRay_GridBins_HH_

#include "MonteRayTypes.hh"
#include "MonteRayVector3D.hh"
#include "ManagedAllocator.hh"

#include "MonteRay_binaryIO.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRayParallelAssistant.hh"
#include "GPUErrorCheck.hh"

#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <fstream>

#include "BinarySearch.hh"

namespace MonteRay {

template <template <class...> class Container>
class MonteRay_GridBins_t : public Managed {

public: // private: TPB TODO: make private
  enum coordinate_t{ LINEAR, RADIAL };
  using Real = gpuRayFloat_t;
  Container<Real> vertices;
  Container<Real> verticesSq;
  coordinate_t type = LINEAR;

  void modifyForRadial(void) {
    type = RADIAL;
  
    // test for negative
    for (const auto& vertex : vertices){
      if (vertex < 0.0){
        throw std::runtime_error("MonteRay_GridBins::modifyForRadial -- vertices must be non-negative !!!");
      }
    }
  
    if( vertices[0] == 0.0 ) {
      vertices.erase(vertices.begin());
    }
  
    // store the vertices values squared
    verticesSq.resize(vertices.size());
    std::transform(vertices.cbegin(), vertices.cend(), verticesSq.begin(), [](auto& val){ return val*val; });
  }

public:
  using Position_t = Vector3D<Real>;
  using Direction_t = Vector3D<Real>;


  MonteRay_GridBins_t() = default;

  MonteRay_GridBins_t(Real min, Real max, int nBins, coordinate_t coord_type = LINEAR) : type(coord_type) {
    if (max < min) { throw std::runtime_error("In constructor MonteRay_GridBins(min, max, nBins) - max is less than min !!!"); }
    if (nBins <= 0) { throw std::runtime_error("In constructor MonteRay_GridBins(min, max, nBins) - nBins is less than or equal to zero !!!"); }
    // for now limit the number of vertices due to fixed memory requirements
    if (nBins >= MAXNUMVERTICES) { throw std::runtime_error("In constructor MonteRay_GridBins(min, max, nBins) - "
       "number of vertices must be less than or equal to " + std::to_string(MAXNUMVERTICES) + " !!!"); }

    vertices.resize(nBins + 1);
    const auto delta = (max-min)/nBins;
    for (int i = 0; i<nBins+1; ++i) {
      vertices[i] = min + i*delta;
    }
    if (type == RADIAL) {
      this->modifyForRadial();
    }
  }

  template<typename OtherContainer>
  MonteRay_GridBins_t(const OtherContainer& otherVertices, coordinate_t coord_type = LINEAR) : 
    vertices(otherVertices.cbegin(), otherVertices.cend()), type(coord_type) 
  {  
    if (vertices.size() >= MAXNUMVERTICES) {
      throw std::runtime_error("In constructor MonteRay_GridBins(vertices) - number of vertices must be less than or equal to " + 
          std::to_string(MAXNUMVERTICES) + " !!!"); 
    }
    if (vertices.size() <= 0) {
      throw std::runtime_error("In constructor MonteRay_GridBins(vertices) - number of vertices must be greater than 0 !!!");
    }

    for( size_t i=1; i < vertices.size(); ++i) {
      if (vertices[i] < vertices[i-1]) { 
        throw std::runtime_error("In constructor MonteRay_GridBins(const Container& vertices) - vertices must be monotonically increasing !!!"); 
      }
    }
    // for now limit the number of vertices due to fixed memory requirements
    MONTERAY_VERIFY( vertices.size() <= MAXNUMVERTICES, "MonteRay_GridBins::setup -- number of vertices exceeds the max size: MAXNUMVERTICES" )
    if (type == RADIAL) {
      this->modifyForRadial();
    }
  }

  std::string className(){ return std::string("MonteRay_GridBins");}

  CUDA_CALLABLE_MEMBER auto numBins() const { 
    return type == RADIAL ?
      vertices.size() : 
      vertices.size() - 1; }
  CUDA_CALLABLE_MEMBER auto getNumBins() const { return numBins(); }
  CUDA_CALLABLE_MEMBER Real getMinVertex() const { return vertices.front(); }
  CUDA_CALLABLE_MEMBER Real getMaxVertex() const { return vertices.back(); }
  CUDA_CALLABLE_MEMBER auto getNumVertices() const { return vertices.size(); }
  CUDA_CALLABLE_MEMBER auto getNumVerticesSq() const { return verticesSq.size(); }
  CUDA_CALLABLE_MEMBER const Real* getVerticesData() const {return vertices.data();}
  CUDA_CALLABLE_MEMBER const Real* getVerticesSqData() const {return verticesSq.data();}
  CUDA_CALLABLE_MEMBER bool isLinear(void) const { return type == LINEAR ? true : false; }
  CUDA_CALLABLE_MEMBER bool isRadial(void) const { return type == RADIAL ? true : false; }

  // returns -1 for one neg side of mesh and number of bins on the pos side of the mesh
  CUDA_CALLABLE_MEMBER inline
  int getLinearIndex(Real pos) const { 
    return pos <= getMinVertex() ?
      -1 :
      pos >= getMaxVertex() ?
        static_cast<int>(getNumBins()) : 
        static_cast<int>(LowerBoundIndex(vertices.data(), vertices.size(), pos));
  }

  CUDA_CALLABLE_MEMBER inline 
  int getRadialIndexFromRSq( Real rSq) const {
    MONTERAY_ASSERT( rSq >= 0.0 );
    return rSq >= verticesSq.back() ? 
      static_cast<int>(getNumBins()) : 
      static_cast<int>(UpperBoundIndex(verticesSq.data(), verticesSq.size(), rSq));
  }

  CUDA_CALLABLE_MEMBER inline 
  int getRadialIndexFromR( Real r) const { return getRadialIndexFromRSq( r*r ); }

  CUDA_CALLABLE_MEMBER inline
  bool isIndexOutside( int i) const { return ( i < 0 ||  i >= getNumBins() ) ? true : false; }

  CUDA_CALLABLE_MEMBER inline
  Real distanceToGetInsideLinearMesh(const Real pos, const Real dir) const {
    Real dist = 0;
    if (pos >= this->vertices.back()){
      dist = dir < 0 ?
        (this->vertices.back() - pos)/dir : 
        std::numeric_limits<Real>::infinity();
    } else if (pos <= this->vertices[0]) {
      dist = dir > 0 ?
        (this->vertices[0] - pos)/dir : 
        std::numeric_limits<Real>::infinity();
    } else {
      return 0.0;
    }
    return dist + std::numeric_limits<Real>::epsilon(); 
  }

#define MR_GRID_BINS_VERSION 1
  void write(std::ostream& outf) const {
    unsigned version = MR_GRID_BINS_VERSION;

    binaryIO::write(outf, version );

    binaryIO::write(outf, vertices.size() );
    for (const auto& vertex : vertices){
      binaryIO::write(outf, vertex);
    }

    binaryIO::write(outf, type );
  }

  static MonteRay_GridBins_t read(std::istream& infile) {
    unsigned version;
    binaryIO::read(infile, version );
    if (version != MR_GRID_BINS_VERSION) {
      throw std::runtime_error("Error while attempting to read MonteRay_GridBins: file version " + 
        std::to_string(version) + " is incompatible with expected version " + std::to_string(MR_GRID_BINS_VERSION));
    }

    size_t nVertices;
    binaryIO::read(infile, nVertices );
    Container<Real> vertices(nVertices);
    for(auto& vertex : vertices){
      binaryIO::read(infile, vertex );
    }

    coordinate_t type;
    binaryIO::read(infile, type );

    return MonteRay_GridBins_t{std::move(vertices), type};
  }

#undef MR_GRID_BINS_VERSION

};

} /* namespace MonteRay */

#include "SimpleVector.hh"
namespace MonteRay{
using MonteRay_GridBins = MonteRay_GridBins_t<SimpleVector>;
}

#endif /* MonteRay_GridBins_HH_ */
