#ifndef SIMPLEVECTOR_H_
#define SIMPLEVECTOR_H_

#include <algorithm>
#include "ManagedAllocator.hh"
#include "detail/HasResize.hpp"

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

  explicit SimpleVector(size_t N) {
    this->resize(N);
  }

  SimpleVector() = default;

  SimpleVector(size_t N, T value): SimpleVector(N) {
    std::fill(begin_, begin_ + size_, value);
  }

  SimpleVector(std::initializer_list<T> init): SimpleVector(init.size()) {
    std::copy(init.begin(), init.end(), begin_);
  }

  template <typename InputIter>
  SimpleVector(InputIter begin, InputIter end): SimpleVector(std::distance(begin,end)) {
    std::copy(begin, end, begin_);
  }

  SimpleVector& operator=(const SimpleVector& other) {
    this->reserve(other.size());
    this->size_ = other.size();
    // copy construct elements in new memory
    // copy assignment is incorrect since element hasn't been constructed (i.e. doesn't exist) yet
    auto it = this->begin();
    for(auto& val : other){
      new (it) T{val};
      it++;
    }
    return *this;
  }

  SimpleVector& operator=(SimpleVector&& other) {
    (*this).~SimpleVector();
    new (this) SimpleVector{std::move(other)};
    return *this;
  }

  SimpleVector(const SimpleVector& other): SimpleVector(other.begin(), other.end()){ }

  SimpleVector(SimpleVector&& other){ 
    this->begin_ = other.begin_;
    this->size_ = other.size_;
    this->reservedSize_ = other.reservedSize_;
    other.begin_ = nullptr;
    other.size_ = 0;
    other.reservedSize_ = 0;
  }

  // generic constructor given another container that has the .resize() method isn't the same container as this one
  template <typename OtherContainer, typename = has_resize_e<OtherContainer>,
            std::enable_if_t< !std::is_same<OtherContainer, SimpleVector<T> >::value, bool > = true >
  SimpleVector(OtherContainer&& other): SimpleVector(other.begin(), other.end()) {}

  ~SimpleVector(){
    if (begin_ != nullptr) {
      for(auto& val : *this) {
        val.~T();
      }
      auto alloc_ = alloc();
      alloc_traits::deallocate(alloc_, begin_, this->capacity());
    }
    size_ = 0;
    reservedSize_ = 0;
  }

  constexpr T const * cbegin() const noexcept { return begin_; }
  constexpr T const * begin() const noexcept { return begin_; }
  constexpr T* begin() noexcept { return begin_; }

  constexpr T* end() noexcept { return begin() + size(); }
  constexpr T const * end() const noexcept { return begin() + size(); }
  constexpr T const * cend() const noexcept { return end(); }

  constexpr T& operator[] (size_t n)  noexcept { return *(this->begin() + n); }
  constexpr const T& operator[] (size_t n) const noexcept { return *(this->begin() + n); }

  constexpr auto size() const noexcept { return size_; }

  constexpr auto capacity() const noexcept { return reservedSize_; }

  void reserve(size_t N) {
    if (N <= this->capacity()){ return; }
    auto alloc_ = alloc();
    auto newBegin = alloc_traits::allocate(alloc_, N);
    if (this->begin() != nullptr){
      auto it = newBegin;
      for (auto& val : *this){
        new (it) T{std::move(val)};
        val.~T(); // destructor in case T's move constructor doesn't actually move T.
        it++;
      }
      alloc_traits::deallocate(alloc_, this->begin(), this->capacity()); 
    }
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

  void erase(T* position){
    erase(position, position + 1);
  }

  void pop_back(){
    erase(this->end() - 1);
  }

  void resizeWithoutConstructing(const size_t N) {
    if (N == size()){
      return;
    } else if (N < size()){ 
      this->erase(begin() + N, end()); 
    } else {
      this->reserve(N);
      // do not default construct all newly allocated elements
      this->size_ = N;
    }
  }

  void resize(const size_t N) {
    if (N == size()){
      return;
    } else if (N < size()){ 
      this->erase(begin() + N, end()); 
    } else {
      this->reserve(N);
      // default construct all newly allocated elements
      for (auto it = this->end(); it < this->begin() + N; it++){
        new (it) T{};
      }
      this->size_ = N;
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

  constexpr T& front() noexcept { return *(this->begin()); }
  constexpr const T& front() const noexcept { return *(this->begin()); }
  constexpr T& back() noexcept { return *(this->end() - 1); }
  constexpr const T& back() const noexcept { return *(this->end() - 1); }

  void clear(){
    for (auto val: *this){
      val.~T();
    }
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

  // note: requires random access iterator
  template <class Iterator>
  void assign(Iterator first, Iterator last){
    this->reserve( std::distance(first, last) );
    this->size_ = 0;
    for (; first != last; first++){
      this->emplace_back(*first); }
  }

  template <typename InputIterator>
  void insert(T* oldPosition, InputIterator&& begin, InputIterator&& end){
    auto N = std::distance(begin, end);
    auto position_dist = std::distance(this->begin(), oldPosition);
    reserve(this->size() + N);
    auto position = this->begin() + position_dist;
    for (auto it = position; it != this->end(); it++){
      *(it + N) = std::move(*it);
    }
    for (; begin != end; begin++){
      *position = *begin;
      position++;
    }
    this->size_ += N;
  }

  constexpr T* data() {
    return begin_;
  }
  constexpr const T* data() const {
    return begin_;
  }
  constexpr bool empty() const {
    return this->size() == 0;
  }

};

} // end namespace MonteRay

#endif
