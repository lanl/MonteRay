#ifndef _MR_HASRESIZE_HH_
#define _MR_HASRESiZE_HH_

#include <type_traits>

namespace MonteRay{

#if __cplusplus < 201703L
  namespace detail {

  template<typename... Ts>
  struct make_void { using type = void; };

  }

  template<typename... Ts>
  using void_t = typename detail::make_void<Ts...>::type;

#else
  template<typename... Ts>
  using void_t = std::void_t<Ts...>;
#endif

template< bool B, class T = void >
using enable_if_t = typename std::enable_if<B,T>::type;

template<typename Container, typename = void > 
  struct HasResizeMethod : std::false_type {};

//Determine if container has resize method
template<typename Container>
struct HasResizeMethod
  <Container, 
  void_t< decltype(std::declval<Container>().resize( std::declval<typename Container::value_type>() ) ) > 
        > : std::true_type {};

template <typename T>
using has_resize_e = enable_if_t<HasResizeMethod<T>::value>;

template <typename T>
using not_has_resize_e = enable_if_t<!HasResizeMethod<T>::value>;

}// end namespace MonteRay

#endif
