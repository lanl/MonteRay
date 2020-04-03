#include "Tally.hh"
#ifdef __CUDACC__
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#endif

namespace MonteRay{

 

void Tally::clear() {
  auto clearFunc = [] CUDADEVICE_CALLABLE_MEMBER (auto&&) { return 0; };
#ifdef __CUDACC__
    thrust::transform(thrust::device, contributions_.begin(), contributions_.end(), contributions_.begin(), clearFunc);
#else
    std::transform(contributions_.begin(), contributions_.end(), contributions_.begin(), clearFunc);
#endif
}

void Tally::accumulate() {
  nSamples_++;
  if (not this->useStats()){
    return; 
  } else {
  auto accumulateFunc = [] CUDADEVICE_CALLABLE_MEMBER (auto&& contribution, auto&& sumAndSumSq) { 
    return MeanAndStdDev{sumAndSumSq.sum() + contribution, sumAndSumSq.sumSq() + contribution*contribution};
  }; 
  auto clearFunc = [] CUDADEVICE_CALLABLE_MEMBER (auto&&) { return 0; };
#ifdef __CUDACC__
    thrust::transform(thrust::device, contributions_.begin(), contributions_.end(), stats_.begin(), stats_.begin(), accumulateFunc);
    thrust::transform(thrust::device, contributions_.begin(), contributions_.end(), contributions_.begin(), clearFunc);
#else
    std::transform(contributions_.begin(), contributions_.end(), stats_.begin(), stats_.begin(), accumulateFunc);
    std::transform(contributions_.begin(), contributions_.end(), contributions_.begin(), clearFunc);
#endif
  }
}

void Tally::computeStats() { 
  if (this->useStats()){
    auto statsFunc = [nSamples_ = this->nSamples_] CUDADEVICE_CALLABLE_MEMBER (auto&& sumAndSumSq){
      auto mean = sumAndSumSq.sum()/nSamples_;
      auto stdDev = nSamples_ > 1 ? Math::sqrt( (sumAndSumSq.sumSq()/nSamples_ - mean*mean)/(nSamples_ - 1) ) : 0;
      return MeanAndStdDev{mean, stdDev};
    };
#ifdef __CUDACC__
    thrust::transform(thrust::device, stats_.begin(), stats_.end(), stats_.begin(), statsFunc);
#else
    std::transform(stats_.begin(), stats_.end(), stats_.begin(), statsFunc);
#endif
  } else {
    auto statsFunc = [nSamples_ = this->nSamples_] CUDADEVICE_CALLABLE_MEMBER (auto&& contribution){
      return contribution/nSamples_;
    };
#ifdef __CUDACC__
    thrust::transform(thrust::device, contributions_.begin(), contributions_.end(), contributions_.begin(), statsFunc);
#else
    std::transform(contributions_.begin(), contributions_.end(), contributions_.begin(), statsFunc);
#endif
  }
}

void Tally::gatherImpl(int mpiRank, const MPI_Comm& mpiComm){
  if (mpiComm != MPI_COMM_NULL){
    if(mpiRank == 0){
      MPI_Reduce(MPI_IN_PLACE, contributions_.data(), contributions_.size(), MPI_DOUBLE, MPI_SUM, 0, mpiComm);
    } else {
      MPI_Reduce( contributions_.data(), nullptr, contributions_.size(), MPI_DOUBLE, MPI_SUM, 0, mpiComm);
      std::memset( contributions_.data(), 0, contributions_.size()*sizeof( TallyFloat ) );
    }
  }
}
  
// Gather inter-work group ranks
void Tally::gather() {
  if( ! MonteRay::isWorkGroupMaster() ) return;
  const auto& PA = MonteRayParallelAssistant::getInstance();
  this->gatherImpl(PA.getInterWorkGroupRank(), PA.getInterWorkGroupCommunicator());
}

// Gather intra-work group ranks 
void Tally::gatherWorkGroup() {
  const auto& PA = MonteRayParallelAssistant::getInstance();
  if( ! PA.isParallel() ) return;
  this->gatherImpl(PA.getWorkGroupRank(), PA.getWorkGroupCommunicator());
}

}
