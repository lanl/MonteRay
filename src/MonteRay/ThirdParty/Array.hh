#ifndef MR_ARRAY_HPP
#define MR_ARRAY_HPP
#include <type_traits>

// This becomes obsolete in C++17.
template <typename T, size_t N>
struct Array
{
  using iterator = T*;
  using const_iterator = const T*;
  using value_type = T;

  value_type data_[N];

  constexpr value_type& operator[](size_t idx) noexcept { return data_[idx]; }
  constexpr const value_type& operator[](size_t idx) const noexcept { return data_[idx]; }

  constexpr T* data() noexcept{ return data_; }
  constexpr const T* data() const noexcept{ return data_; }
  
  constexpr size_t size() const noexcept { return N; }

  constexpr iterator begin() noexcept{ return data(); }
  constexpr const_iterator begin() const noexcept{ return data(); }
  constexpr iterator end() noexcept{ return data() + N; }
  constexpr const_iterator end() const noexcept{ return data() + N; }
  constexpr const_iterator cbegin() const noexcept{ return data(); }
  constexpr const_iterator cend() const noexcept{ return data() + N; }
};

namespace details {
  template <class D, class... Types>
  using make_array_return_type = Array<typename std::common_type<Types...>::type,
                                 sizeof...(Types)>;
}
 
template < class D = void, class... Types>
constexpr details::make_array_return_type<D, Types...> make_array(Types&&... t) {
  return {std::forward<Types>(t)... };
}

#endif 
