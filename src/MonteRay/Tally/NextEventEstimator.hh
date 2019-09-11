#ifndef MONTERAYNEXTEVENTESTIMATOR_HH_
#define MONTERAYNEXTEVENTESTIMATOR_HH_

#define NEE_VERSION static_cast<unsigned>(1)

#include <iostream>
#include <sys/types.h>
#include <unistd.h>
#include <limits>
#include <tuple>

#include "MonteRayTypes.hh"
#include "MonteRayAssert.hh"
#include "MonteRayVector3D.hh"

#include "ManagedAllocator.hh"

#include "Tally.hh"
#include "RayWorkInfo.hh"

#include "CrossSection.hh"

namespace MonteRay {

template< unsigned N>
class RayList_t;
template< unsigned N>
class Ray_t;

class NextEventEstimator: public Managed {
public:
  using tally_t = gpuTallyType_t;
  using position_t = gpuRayFloat_t;
  using DetectorIndex_t = int;
private:
  SimpleVector<Vector3D<position_t>> tallyPoints_;
  SimpleVector<gpuFloatType_t> tallyTimeBinEdges_;
  Tally tally_;
  position_t radius_;

  NextEventEstimator(SimpleVector<Vector3D<position_t>>&& tallyPoints, SimpleVector<gpuFloatType_t>&& tallyTimeBinEdges, gpuFloatType_t radius):
    tallyPoints_(std::move(tallyPoints)), 
    tallyTimeBinEdges_(std::move(tallyTimeBinEdges)),
    tally_(tallyPoints_.size(), tallyTimeBinEdges_),
    radius_(radius)
  { }

public:
  NextEventEstimator() = default; // TPB TODO: used by RayListController - remove need for this
  class Builder{
    private:
    SimpleVector<Vector3D<position_t>> b_tallyPoints_;
    SimpleVector<gpuFloatType_t> b_tallyTimeBinEdges_;
    position_t b_radius_;

    public:
    Builder(int num = 1){
      b_tallyPoints_.reserve(num);
    }

    auto addTallyPoint( position_t x, position_t y, position_t z){
       b_tallyPoints_.emplace_back(x, y, z);
       return b_tallyPoints_.size() - 1; // RayListController expects an index of the placed point back.
    }

    auto add( position_t x, position_t y, position_t z){ // TODO: deprecate
       return this->addTallyPoint(x, y, z);
    }

    void setExclusionRadius(position_t r) { 
      printf("Warning: MonteRay::NextEventEstimator Exclusion radius is not yet implemented.\n");
      b_radius_ = r; }

    template<typename Edges>
    void setTimeBinEdges( const Edges& edges) {
      b_tallyTimeBinEdges_.assign(edges.begin(), edges.end());
    }

    auto build() {
      // TODO: resize tallyPoints, i.e. implement shrink_to_fit for SimpleVector
      return NextEventEstimator(std::move(b_tallyPoints_), std::move(b_tallyTimeBinEdges_), b_radius_);
    }

    template<typename Stream>
    auto read(Stream& in){
      unsigned version;
      binaryIO::read( in, version );
      if (version != NEE_VERSION){
        throw std::runtime_error("NextEventEstimator dump file version number " + 
            std::to_string(version) + " is incompatible with expected version " + 
            std::to_string(NEE_VERSION));
      }
      size_t size;
      binaryIO::read( in, size );
      b_tallyPoints_.resizeWithoutConstructing(size);
      for (auto& tallyPoint : b_tallyPoints_){
        binaryIO::read( in, tallyPoint[0] );
        binaryIO::read( in, tallyPoint[1] );
        binaryIO::read( in, tallyPoint[2] );
      }
      binaryIO::read( in, size );
      b_tallyTimeBinEdges_.resizeWithoutConstructing(size);
      for (auto& val : b_tallyTimeBinEdges_){
        binaryIO::read( in, val );
      }
      binaryIO::read( in, b_radius_ );
      return NextEventEstimator(std::move(b_tallyPoints_), std::move(b_tallyTimeBinEdges_), b_radius_);
    }
  };

  CUDA_CALLABLE_MEMBER int size(void) const { return tallyPoints_.size(); }

  CUDA_CALLABLE_MEMBER position_t getExclusionRadius(void) const { return radius_; }

  CUDA_CALLABLE_MEMBER tally_t getTally(int spatialIndex, int timeIndex=0) const {
    return tally_.getTally(spatialIndex, timeIndex);
  }

  template<unsigned N, typename Geometry, typename MaterialProperties, typename MaterialList>
  CUDA_CALLABLE_MEMBER tally_t calcScore( const int threadID, const Ray_t<N>& ray, RayWorkInfo& rayInfo, 
      const Geometry& geometry, const MaterialProperties& matProps, const MaterialList& matList);

  template<unsigned N, typename Geometry, typename MaterialProperties, typename MaterialList>
  void cpuScoreRayList( const RayList_t<N>* pRayList, RayWorkInfo* pRayInfo, 
      const Geometry* const pGeometry, const MaterialProperties* const pMatProps, const MaterialList* const pMatList){
    for(auto particleID = 0; particleID < pRayList->size(); particleID++) {
      constexpr int threadID = 0;
      pRayInfo->clear(threadID);
      auto& ray = pRayList->points[particleID];
      calcScore(threadID, ray, *pRayInfo, *pGeometry, *pMatProps, *pMatList);
    }
  }

  template<unsigned N, typename Geometry, typename MaterialProperties, typename MaterialList>
  void launch_ScoreRayList( int nBlocks, int nThreads, const RayList_t<N>* pRayList, RayWorkInfo* pRayInfo, 
      const Geometry* const pGeometry, const MaterialProperties* const pMatProps, const MaterialList* const pMatList, 
      const cudaStream_t* const stream);

  const auto& getPoint(int i) const { 
    MONTERAY_ASSERT(i<tallyPoints_.size());  
    return tallyPoints_[i]; 
  }

  void printPointDets( const std::string& outputFile, int nSamples, int constantDimension=2){
    if( MonteRayParallelAssistant::getInstance().getWorkGroupRank() != 0 ) return;

    if( tallyPoints_.size() == 0 ) {
        return;
    }

    std::ofstream out;
    out.open(outputFile.c_str(), std::ios::out );
    if( ! out.is_open() ) {
        throw std::runtime_error( "Failure opening output file.  File= " + outputFile );
    }
    outputTimeBinnedTotal( out, nSamples, constantDimension);
    out.close();
  }

  void outputTimeBinnedTotal(std::ostream& out, int nSamples=1, int constantDimension=2){
    out << "#  MonteRay results                                                    \n";
    out << "#                                                                      \n";
    out << "#       X          Y          Z      Time        Score        Score    \n";
    out << "#     (cm)       (cm)       (cm)   (shakes)      Average      Rel Err. \n";
    out << "# ________   ________   ________   ________   ___________   ___________ \n";

    gpuFloatType_t sum = 0.0;
    gpuFloatType_t min = std::numeric_limits<double>::infinity();
    gpuFloatType_t max = -std::numeric_limits<double>::infinity();

    // dim2 used to insert new-line when it decreases, indicating a new row.
    unsigned dim2;
    switch (constantDimension) {
    case 0:
        dim2 = 2; // z
        break;
    case 1:
        dim2 = 2; // z
        break;
    case 2:
        dim2 = 1; // y
        break;
    default:
        break;
    }

    Vector3D<gpuFloatType_t> pos = getPoint(0);

    // previousSecondDimPosition used to detect when to insert carriage return
    double previousSecondDimPosition = pos[dim2];

    for( unsigned i=0; i < tallyPoints_.size(); ++i ) {
        for( int j=0; j < tally_.getNumTimeBins(); ++j ) {

            double time = tally_.getTimeBinEdge(j);
            Vector3D<gpuFloatType_t> pos = getPoint(i);
            gpuFloatType_t value = getTally(i,j) / nSamples;

            if(  pos[dim2] < previousSecondDimPosition ) {
                out << "\n";
            }

            char buffer[200];
            snprintf( buffer, 200, "  %8.3f   %8.3f   %8.3f   %8.3f   %11.4e   %11.4e\n",
                                     pos[0], pos[1], pos[2],   time,   value,     0.0 );
            out << buffer;

            previousSecondDimPosition = pos[dim2];

            if( value < min ) min = value;
            if( value > max ) max = value;
            sum += value;
        }
    }
    out << "\n#\n";
    out << "# Min value = " << min << "\n";
    out << "# Max value = " << max << "\n";
    out << "# Average value = " << sum / tallyPoints_.size() << "\n";
  }

  const auto& getTimeBinEdges() const {
    return tallyTimeBinEdges_;
  }

  void gather(){
    tally_.gather();
  }

  // gather work group is rarely used, mainly for testing
  void gatherWorkGroup(){
    tally_.gatherWorkGroup();
  }

  template<typename Stream>
  void write(Stream& out){
    binaryIO::write( out, NEE_VERSION);
    binaryIO::write( out, tallyPoints_.size() );
    for (auto& tallyPoint : tallyPoints_){
       binaryIO::write( out, tallyPoint[0] );
       binaryIO::write( out, tallyPoint[1] );
       binaryIO::write( out, tallyPoint[2] );
    }
    binaryIO::write( out, tallyTimeBinEdges_.size() );
    for (auto& val : tallyTimeBinEdges_){
      binaryIO::write( out, val );

    }
    binaryIO::write( out, radius_ );
  }

};

template<unsigned N, typename Geometry, typename MaterialProperties, typename MaterialList>
CUDA_CALLABLE_KERNEL  kernel_ScoreRayList(NextEventEstimator* ptr, const RayList_t<N>* pRayList, 
      const Geometry* const pGeometry, const MaterialProperties* const pMatProps, const MaterialList* const pMatList);

} /* namespace MonteRay */

#undef NEE_VERSION
#endif /* MONTERAYNEXTEVENTESTIMATOR_HH_ */
