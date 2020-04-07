#ifndef CUDAWRAPPERS_HH_
#define CUDAWRAPPERS_HH_

#include <memory>

#ifndef __CUDACC__
typedef int cudaStream_t;
typedef int cudaEvent_t;
#endif

namespace MonteRay{
namespace cuda{
  class StreamPointer{
    private: 
    std::shared_ptr<cudaStream_t> pStream_;

    public:
    StreamPointer();
    auto& operator=(const StreamPointer& other) {pStream_ = other.pStream_; return *this;}
    auto& operator=(StreamPointer&& other) {pStream_ = std::move(other.pStream_); return *this;}
    StreamPointer(const StreamPointer& other): pStream_(other.pStream_) {}
    StreamPointer(StreamPointer&& other): pStream_(std::move(other.pStream_)) {}
    ~StreamPointer();
    auto get() { return pStream_.get(); }
    auto get() const { return pStream_.get(); }
    void swap(StreamPointer& other) { pStream_.swap(other.pStream_); }
    auto& operator*() { return *pStream_; }
    const auto& operator*() const { return *pStream_; }
  };

  class EventPointer{
    private: 
    std::shared_ptr<cudaEvent_t> pEvent_;

    public:
    EventPointer();
    auto& operator=(const EventPointer& other) {pEvent_ = other.pEvent_; return *this;}
    auto& operator=(EventPointer&& other) {pEvent_ = std::move(other.pEvent_); return *this;}
    EventPointer(const EventPointer& other): pEvent_(other.pEvent_) {}
    EventPointer(EventPointer&& other): pEvent_(std::move(other.pEvent_)) {}
    auto get() { return pEvent_.get(); }
    auto get() const { return pEvent_.get(); }
    void swap(EventPointer& other) { pEvent_.swap(other.pEvent_); }
    auto& operator*() { return *pEvent_; }
    const auto& operator*() const { return *pEvent_; }
    ~EventPointer();
  };
} // end namespace cuda
} // end namespace MonteRay

#endif
