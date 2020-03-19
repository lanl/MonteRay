#ifndef MONTERAY_TALLY_HH_
#define MONTERAY_TALLY_HH_

#include <vector>
#include <cstring>
#include <string>
#include <memory>

#include "MonteRayConstants.hh"
#include "MonteRayParallelAssistant.hh"
#include "MonteRayAssert.hh"
#include "Containers.hh"

namespace MonteRay {

class Tally : public Managed {

private:
  int numSpatialBins_ = 1;
  SimpleView<gpuFloatType_t> timeBinEdges_;
  SimpleVector<gpuTallyType_t> data_;

public:
  template <typename TimeEdges>
  Tally(int numSpatialBins, TimeEdges&& timeEdges) : numSpatialBins_(numSpatialBins), timeBinEdges_(timeEdges) {
    if( numSpatialBins <= 0 ) {
      throw std::runtime_error("Attempting to set number of spatial bins in a tally to invalid value: " + std::to_string(numSpatialBins));
    }
    auto data_size = getIndex( (numSpatialBins_-1), timeBinEdges_.size() ) + 1;
    data_.resize(data_size);
  }

  Tally(int numSpatialBins=1) : Tally(numSpatialBins, SimpleView<gpuFloatType_t>{})  { }

  CUDA_CALLABLE_MEMBER int numSpatialBins() const {
      return numSpatialBins_;
  }

  template<typename Edges> // TODO: make edges const
  void setTimeBinEdges(const Edges& edges) {
    new (&timeBinEdges_) SimpleView<gpuFloatType_t>(edges);
  }

  CUDA_CALLABLE_MEMBER
  int getNumTimeBins() const {
      return timeBinEdges_.size() + 1;
  }

  gpuFloatType_t getTimeBinEdge(int i ) const {
    return timeBinEdges_.size() != 0 ?  
      timeBinEdges_[i] : 
      std::numeric_limits<gpuFloatType_t>::max(); 
  }

  CUDA_CALLABLE_MEMBER
  int getIndex(int spatial_index, int time_index) const {
    // layout is first by spatial index, then by time index
    // assuming threads are contributing to the same time index
    return  time_index*numSpatialBins_ + spatial_index;
  }

  CUDA_CALLABLE_MEMBER
  int getTimeIndex( const gpuFloatType_t time ) const {
    // TODO: test linear search (done here) vs. binary search
    // this automatically takes care of non-infinite right edges.
    int index = 0;
    //for (const auto& val : timeBinEdges_){
    for (unsigned i=0; i < timeBinEdges_.size(); ++i ){
      if (time < timeBinEdges_[i]){
        return index;
      }
      index++;
    }
    return index;
  }

  CUDA_CALLABLE_MEMBER
  void scoreByIndex(gpuTallyType_t value, int spatial_index, int time_index=0 );

  CUDA_CALLABLE_MEMBER
  void score( gpuTallyType_t value, int spatial_index, gpuFloatType_t time = 0.0 ){
    int time_index = getTimeIndex( time );
    scoreByIndex( value, spatial_index, time_index );
  }

  CUDA_CALLABLE_MEMBER
  gpuTallyType_t getTally(int spatial_index, int time_index = 0 ) const {
    int index = getIndex(spatial_index, time_index);
    MONTERAY_ASSERT(data_.size() > index);
    return data_[ index ];
  }

  CUDA_CALLABLE_MEMBER
  int getTallySize() const {
    return data_.size();
  }

  void setupForParallel() {} // TODO: remove this ?

  void gather(){
    if( ! MonteRay::isWorkGroupMaster() ) return;

    SimpleVector<gpuTallyType_t> globalData;

    if( MonteRayParallelAssistant::getInstance().getInterWorkGroupRank() == 0 ) {
      globalData.resize(data_.size());
    }

    if( MonteRayParallelAssistant::getInstance().getInterWorkGroupCommunicator() != MPI_COMM_NULL ) {
        MPI_Reduce( data_.begin(), globalData.begin(), data_.size(), MPI_DOUBLE, MPI_SUM, 0, MonteRayParallelAssistant::getInstance().getInterWorkGroupCommunicator());
    }

    if( MonteRayParallelAssistant::getInstance().getInterWorkGroupRank() != 0 ) {
      std::memset( data_.begin(), 0, data_.size()*sizeof( gpuTallyType_t ) );
    } else {
        data_ = std::move(globalData);
    }
  }

  // Used mainly for testing
  void gatherWorkGroup(){
    // For testing - setup like gather but allows direct scoring on all ranks of the work group
    if( ! MonteRayParallelAssistant::getInstance().isParallel() ) return;
    SimpleVector<gpuTallyType_t> globalData;

    if( MonteRayParallelAssistant::getInstance().getWorkGroupRank() == 0 ) {
      globalData.resize(data_.size());
    }

    if( MonteRayParallelAssistant::getInstance().getWorkGroupCommunicator() != MPI_COMM_NULL ) {
        MPI_Reduce( data_.begin(), globalData.begin(), data_.size(), MPI_DOUBLE, MPI_SUM, 0, MonteRayParallelAssistant::getInstance().getWorkGroupCommunicator());
    }

    if( MonteRayParallelAssistant::getInstance().getWorkGroupRank() != 0 ) {
      std::memset( data_.begin(), 0, data_.size()*sizeof( gpuTallyType_t ) );
    } else {
        data_ = std::move(globalData);
    }
  }
};

} // end namespace

#endif /* MONTERAYTALLY_HH_ */
