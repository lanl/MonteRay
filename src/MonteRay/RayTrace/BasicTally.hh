#ifndef MR_GPUTALLY_H_
#define MR_GPUTALLY_H_

#include "ManagedAllocator.hh"
#include "GPUAtomicAdd.hh"
#include "SimpleVector.hh"
#include "MonteRayAssert.hh"
#include "MonteRay_binaryIO.hh"

namespace MonteRay {

class BasicTally : public Managed {

  private:
  SimpleVector<gpuTallyType_t> tally_;

  public:
  BasicTally(size_t N) : tally_(N, 0){ }

  BasicTally(SimpleVector<gpuTallyType_t>&& tally) : tally_(std::move(tally)){ }

  void clear() { 
    for (auto& val : tally_){
      val = 0;
    }
  }

  CUDA_CALLABLE_MEMBER
  auto size() const { return tally_.size(); }
  const auto& getTallies() { return tally_; };
  auto data() { return tally_.data(); };

  CUDA_CALLABLE_MEMBER
  void scoreByIndex(int index, gpuTallyType_t value){
    MONTERAY_ASSERT(index < this->size());
    gpu_atomicAdd( &(tally_[index]), value);
  }

  template< typename Particle>
  CUDA_CALLABLE_MEMBER
  void score(const Particle& particle, gpuTallyType_t value) {
    scoreByIndex(particle.getDetectorIndex(), value);
  }


  CUDA_CALLABLE_MEMBER 
  auto getTally(int index) const { return tally_[index]; }

#define MR_BASICTALLY_VERSION 0
  void write(std::ostream& stream) {
    unsigned version = MR_BASICTALLY_VERSION;
    binaryIO::write(stream,version);
    unsigned size = this->size();
    binaryIO::write(stream, size);
    for (const auto& val : tally_){
      binaryIO::write(stream, val);
    }
  }

  static BasicTally read(std::istream& stream) {
    unsigned version;
    binaryIO::read(stream,version);
    if (version != MR_BASICTALLY_VERSION) {
      throw std::runtime_error("BasicTally::read is attempting to read a binary file with "
          " version " + std::to_string(version) + " but is expecting version " + 
          std::to_string(MR_BASICTALLY_VERSION));
    }

    unsigned size;
    binaryIO::read(stream,size);

    SimpleVector<gpuTallyType_t> tallyData(size);

    for (auto& val : tallyData){
        binaryIO::read(stream, val);
    }
    return BasicTally(std::move(tallyData));
  }
#undef MR_BASICTALLY_VERSION

};

} // end namespace MonteRay

#endif
