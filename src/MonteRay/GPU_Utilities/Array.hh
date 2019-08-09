#ifndef MONTERAY_ARRAY_HPP
#define MONTERAY_ARRAY_HPP
#include <array>

namespace MonteRay {
// This becomes obsolete in C++17.
template <typename T, size_t N>
class Array
{
    
  public:
  T data_[N ? N : 1];
  using iterator = typename std::array<T, N>::iterator;
  using value_type = T;

  constexpr value_type& operator[](size_t idx) noexcept {
    return data_[idx];
  }

  constexpr const value_type& operator[](size_t idx) const noexcept {
    return data_[idx];
  }
  
  constexpr T* data() noexcept{
    return data_;
  }

  constexpr size_t size() const noexcept { return N; }

  constexpr iterator begin() noexcept{
    return data();
  }

  constexpr iterator end() noexcept{
    return data() + N;
  }

};

}

#endif 
