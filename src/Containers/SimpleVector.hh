#ifndef SIMPLEVECTOR_H_
#define SIMPLEVECTOR_H_

#include <algorithm>

namespace MonteRay{

// A cheap, partial implementation of std::vector.
template <class T, class alloc>
class SimpleVector 
{
  private:
    using alloc_traits = std::allocator_traits<alloc>;
    T* begin_ = nullptr;
    size_t size_ = 0;
  public:

  explicit SimpleVector(size_t count): size_(count) {
    auto alloc_ = alloc();
    begin_ = alloc_traits::allocate(alloc_, count);
  }

  SimpleVector() = default;

  SimpleVector(size_t count, T value): SimpleVector(count) {
    std::fill(begin_, begin_ + size_, value);
  }

  SimpleVector(std::initializer_list<T> init): SimpleVector(init.size()) {
    std::copy(init.begin(), init.end(), begin_);
  }

  SimpleVector(const T* const begin, const T* const end): SimpleVector(std::distance(begin,end)) {
    std::copy(begin, end, begin_);
  }

  SimpleVector& operator=(const SimpleVector& other) {
    this->resize(other.size());
    std::copy(other.begin(), other.end(), this->begin_);
    return *this;
  }

  SimpleVector& operator=(SimpleVector&& other) {
    auto alloc_ = alloc();
    begin_ = alloc_traits::allocate(alloc_, other.size());
    this->begin_ = other.begin_;
    this->size_ = other.size_;
    other.begin_ = nullptr;
    other.size_ = 0;
    return *this;
  }

  SimpleVector(const SimpleVector& other): SimpleVector(other.begin(), other.end()){ }

  SimpleVector(SimpleVector&& other){ 
    *this = std::move(other); 
  }

  ~SimpleVector(){
    auto alloc_ = alloc();
    if (begin_ != nullptr) alloc_traits::deallocate(alloc_, begin_, this->size());
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

  void resize(size_t count) {
    if (count == size_) return;
    auto alloc_ = alloc();
    if (begin_ != nullptr) alloc_traits::deallocate(alloc_, begin_, this->size());
    if (count > 0) begin_ = alloc_traits::allocate(alloc_, count);
    size_ = count;
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
    other.begin_ = this->begin_;
    other.size_ = this->size_;
    this->begin_ = tempBegin;
    this->size_ = tempSize;
  }

};

} // end namespace MonteRay
#endif
