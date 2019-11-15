#ifndef MR_ARRAY_HPP
#define MR_ARRAY_HPP
#include <array>
#include <type_traits>


// This becomes obsolete in C++17.
template <typename T, size_t N>
class Array: public std::array<T, N>
{
  private:
    using base_t = std::array<T, N>;
    using value_type = T;
#ifdef __GLIBCXX__
    using base_t::_M_elems;
#endif
#ifdef _LIBCPP_VERSION
    using base_t::__elems_;
#endif
  public:
  constexpr Array(): base_t{} {};

  // nonsense to please nvcc - some mysterious constructor is called otherwise (in device code)
  template<typename Arg = Array, typename... Args, typename U = T, std::enable_if_t<not std::is_fundamental<U>::value, bool> = true > //sfinae here>
  constexpr Array(Arg&& arg, Args&&... args): base_t{std::forward<Arg>(arg), std::forward<Args>(args)...}{ }

  // nonsense to please nvcc - some mysterious constructor is called otherwise (in device code)
  template<typename Arg = Array, typename... Args, typename U = T, std::enable_if_t<std::is_fundamental<U>::value, bool> = true > //sfinae here>
  constexpr Array(Arg&& arg, Args&&... args): base_t{std::forward<Arg>(arg), std::forward<Args>(args)...}{ 
    volatile T t=(*this)[0]; // just makes a temporary throw-away value so NVCC doesn't complain 
    (*this)[0]=t;
  }
  
#ifdef __GLIBCXX__
  constexpr value_type& operator[](size_t idx) noexcept {
    return _M_elems[idx];
  }
  constexpr const value_type& operator[](size_t idx) const noexcept {
    return _M_elems[idx];
  }
  constexpr T* data() noexcept{
    return _M_elems;
  }

  constexpr const T* data() const noexcept{
    return _M_elems;
  }
#endif
#ifdef _LIBCPP_VERSION
  constexpr value_type& operator[](size_t idx) noexcept {
    return __elems_[idx];
  }
  constexpr const value_type& operator[](size_t idx) const noexcept {
    return __elems_[idx];
  }
  constexpr T* data() noexcept{
    return __elems_;
  }
  constexpr const T* data() const noexcept{
    return __elems_;
  }
#endif
  
  constexpr size_t size() const noexcept { return N; }

  constexpr typename base_t::iterator begin() noexcept{
    return data();
  }

  constexpr typename base_t::const_iterator begin() const noexcept{
    return data();
  }

  constexpr typename base_t::iterator end() noexcept{
    return data() + N;
  }

  constexpr typename base_t::const_iterator end() const noexcept{
    return data() + N;
  }

};

namespace details {
 
  template <class D, class... Types>
  using return_type = std::array<typename std::common_type<Types...>::type,
                                 sizeof...(Types)>;

  template <class D, class... Types>
  using shacl_return_type = Array<typename std::common_type<Types...>::type,
                                 sizeof...(Types)>;
}
 
template < class D = void, class... Types>
constexpr details::shacl_return_type<D, Types...> make_array(Types&&... t) {
  return {std::forward<Types>(t)... };
}

#endif 
