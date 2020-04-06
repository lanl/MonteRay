#include "StreamAndEvent.hh"

namespace MonteRay{
namespace cuda{
StreamPointer::StreamPointer(): pStream_(std::make_shared<cudaStream_t>()) {
#ifdef __CUDACC__
  cudaStreamCreate(pStream_.get());
#else
  *pStream_ = 0;
#endif
}

StreamPointer::~StreamPointer() {
#ifdef __CUDACC__
  if (pStream_.use_count() == 1) {
    cudaStreamDestroy(*pStream_);
  }
#endif
}

EventPointer::~EventPointer(){
#ifdef __CUDACC__
  if (pEvent_.use_count() == 1) {
    cudaEventDestroy(*pEvent_);
  }
#endif
}

EventPointer::EventPointer(): pEvent_(std::make_shared<cudaEvent_t>()) {
#ifdef __CUDACC__
  cudaEventCreate(pEvent_.get());
#else
  *pEvent_ = 0;
#endif
}
}
}
