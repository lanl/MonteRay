#ifndef MR_FILTER_H_
#define MR_FILTER_H_

#include "MonteRay_binaryIO.hh"
 
#define MR_FILTER_VERSION static_cast<unsigned>(2)

namespace MonteRay {

template <class Container>
class BinEdgeFilter{
  private:
  Container binEdges_;
  public:

  BinEdgeFilter() = default;
  template <typename OtherContainer, 
           std::enable_if_t< !std::is_same<Container, BinEdgeFilter<Container> >::value, bool > = true >
  BinEdgeFilter(OtherContainer&& binEdges): binEdges_(std::move(binEdges)) {}

  const auto& binEdges() const { return binEdges_; }
  CUDA_CALLABLE_MEMBER auto size() const { return binEdges_.size() + 1; }

  template <typename T>
  CUDA_CALLABLE_MEMBER auto operator()(T&& val) const {  // linear search
    int index = 0;
    for (const auto& binEdge : binEdges_){
      if (val < binEdge) {
        return index;
      } else { 
        index++;
      }
    }
    return index;
  };

  template<typename Stream>
  void write(Stream& out){
    binaryIO::write(out, MR_FILTER_VERSION);
    binaryIO::write(out, binEdges_);
  }

  template <typename Stream>
  static auto read(Stream& in){
    unsigned version;
    binaryIO::read( in, version );
    if (version != MR_FILTER_VERSION){
      throw std::runtime_error("Filter file version number " + 
          std::to_string(version) + " is incompatible with expected version " + 
          std::to_string(MR_FILTER_VERSION));
    }

    Container binEdges;
    binaryIO::read(in, binEdges);
    return BinEdgeFilter(std::move(binEdges));
  }

};

}// end namespace MonteRay

#undef MR_FILTER_VERSION 

#endif
