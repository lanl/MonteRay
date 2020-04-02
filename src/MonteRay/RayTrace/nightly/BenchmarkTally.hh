#ifndef MR_GPUTALLY_H_
#define MR_GPUTALLY_H_

#include "MonteRay_binaryIO.hh"

namespace MonteRay {

// class used to read in benchmark tally data for tests
class BenchmarkTally {
  private:
  std::vector<gpuTallyType_t> tally_;

  public:
  BenchmarkTally(std::vector<gpuTallyType_t> tally): tally_(std::move(tally)) {}

  auto size() const { return tally_.size(); }

  auto operator[](size_t i) {return tally_[i];}
  const auto& data() {return tally_;}

#define MR_BASICTALLY_VERSION 0
  static BenchmarkTally read(std::istream& stream) {
    unsigned version;
    binaryIO::read(stream,version);
    if (version != MR_BASICTALLY_VERSION) {
      throw std::runtime_error("BasicTally::read is attempting to read a binary file with "
          " version " + std::to_string(version) + " but is expecting version " + 
          std::to_string(MR_BASICTALLY_VERSION));
    }

    unsigned size;
    binaryIO::read(stream,size);

    std::vector<gpuTallyType_t> tallyData(size);

    for (auto& val : tallyData){
        binaryIO::read(stream, val);
    }
    return {std::move(tallyData)};
  }
#undef MR_BASICTALLY_VERSION

};

} // end namespace MonteRay

#endif
