#ifndef SIMPLEVECTOR_H_
#define SIMPLEVECTOR_H_

#include <algorithm>
#include "ManagedAllocator.hh"

namespace MonteRay{

template <class T, class alloc=managed_allocator<T>>
class SimpleVector : public Managed
{
  private:
    using alloc_traits = std::allocator_traits<alloc>;
    T* begin_ = nullptr;
    size_t size_ = 0;
    size_t reservedSize_ = 0;
  public:

  explicit SimpleVector(size_t N): size_(N), reservedSize_(N) {
    auto alloc_ = alloc();
    begin_ = alloc_traits::allocate(alloc_, N);
  }

  SimpleVector() = default;

  SimpleVector(size_t N, T value): SimpleVector(N) {
    std::fill(begin_, begin_ + size_, value);
  }

  SimpleVector(std::initializer_list<T> init): SimpleVector(init.size()) {
    std::copy(init.begin(), init.end(), begin_);
  }

  SimpleVector(const T* const begin, const T* const end): SimpleVector(std::distance(begin,end)) {
    std::copy(begin, end, begin_);
  }

  SimpleVector& operator=(const SimpleVector& other) {
    this->reserve(other.capacity());
    std::copy(other.begin(), other.end(), this->begin_);
    return *this;
  }

  SimpleVector& operator=(SimpleVector&& other) {
    auto alloc_ = alloc();
    begin_ = alloc_traits::allocate(alloc_, other.size());
    this->begin_ = other.begin_;
    this->size_ = other.size_;
    this->reservedSize_ = other.reservedSize_;
    other.begin_ = nullptr;
    other.size_ = 0;
    other.reservedSize_ = 0;
    return *this;
  }

  SimpleVector(const SimpleVector& other): SimpleVector(other.begin(), other.end()){ }

  SimpleVector(SimpleVector&& other){ 
    *this = std::move(other); 
  }

  ~SimpleVector(){
    auto alloc_ = alloc();
    if (begin_ != nullptr) alloc_traits::deallocate(alloc_, begin_, this->capacity());
  }

  constexpr T const * begin() const noexcept {
    return begin_;
  }

  constexpr T* begin() noexcept {
    return begin_;
  }

  constexpr T const * end() const noexcept {
    return begin() + size();
  }

  constexpr T* end() noexcept {
    return begin() + size();
  }

  constexpr T& operator[] (size_t n)  noexcept {
    return *(this->begin() + n);
  }

  constexpr const T& operator[] (size_t n) const noexcept {
    return *(this->begin() + n);
  }

  constexpr auto size() const noexcept { 
    return size_;
  }

  constexpr auto capacity() const noexcept { 
    return reservedSize_;
  }

  void reserve(size_t N) {
    if (N <= this->capacity()){ return; }
    auto alloc_ = alloc();
    auto newBegin = alloc_traits::allocate(alloc_, N);
    std::copy(begin(), end(), newBegin);
    if (begin_ != nullptr){ alloc_traits::deallocate(alloc_, begin_, this->size()); }
    this->begin_ = newBegin;
    this->reservedSize_ = N;
  }

  void erase(T* start, T* finish){
    std::for_each(start, finish, [](T& val){val.~T();});
    if (finish != this->end()) {
      std::move(finish, this->end(), start);
    }
    this->size_ -= std::distance(start, finish);
  }

  void resize(const size_t N) {
    if (N == size()){
      return;
    } else if (N < size()){ 
      this->erase(begin() + N, end()); 
    } else {
      this->reserve(N);
      this->size_ = capacity();
    }
  }

  template<typename... Args>
  T& emplace_back(Args&&... args){
    if (size() == capacity()){
      capacity() == 0 ? reserve(1) : reserve(2*capacity());
    }
    T* pRetval = new ( end() ) T(std::forward<Args>(args)...);
    size_++;
    return *pRetval;
  }

  void push_back(const T& val){
    this->emplace_back(val);
  }

  constexpr T& back() noexcept { 
    return *(this->end() - 1);
  }

  void clear(){
    auto alloc_ = alloc();
    if (begin_ != nullptr) alloc_traits::deallocate(alloc_, begin_, this->size());
    size_ = 0;
  }

  void swap(SimpleVector& other){
    auto tempBegin = other.begin_;
    auto tempSize = other.size_;
    auto tempReservedSize = other.reservedSize_;
    other.begin_ = this->begin_;
    other.size_ = this->size_;
    other.reservedSize_= this->reservedSize_;
    this->begin_ = tempBegin;
    this->size_ = tempSize;
    this->reservedSize_ = tempReservedSize;
  }

};

} // end namespace MonteRay

#endif
