#ifndef SIMPLEVIEW_H_
#define SIMPLEVIEW_H_

namespace MonteRay{
template <class T>
class SimpleView
{
  private:
  T* begin_ = nullptr;
  T* end_ = nullptr;

  public:
  SimpleView() {} // creates a view of nothing
  SimpleView(const T* const begin, const T* const end): begin_(begin), end_(end) {}

  template <typename Iterator>
  SimpleView(Iterator&&  begin, Iterator&& end): begin_(&(*begin)), end_(&(*end)) {}
  
  // sfinae to get back the default copy/move ctors and assignment operators
  template <typename Container, 
           std::enable_if_t< 
             !std::is_same<Container,SimpleView>::value, 
             bool
           > = true >
  constexpr SimpleView(Container&& container): begin_(&(*container.begin())), end_(&(*container.end())){ }

  constexpr auto& operator[](size_t i) noexcept { return *(begin_ + i);}
  constexpr const auto& operator[](size_t i) const noexcept { return *(begin_ + i);}

  constexpr auto begin() { return begin_; }
  constexpr const auto begin() const { return begin_; }
  constexpr auto end() { return end_; }
  constexpr const auto end() const { return end_; }
  constexpr size_t size() const { return end_ - begin_; }

  template <typename Container>
  constexpr bool operator== (const Container& other) const {
    return ( (this->begin() == other.begin()) and (this->end() == other.end()) );
  }

  template <typename Container>
  constexpr bool operator!= (const Container& other) const {
    return  (not this->operator==(other));
  }

};

template <typename Iterator>
auto make_simple_view(Iterator&& begin, Iterator&& end){
  using T = std::remove_reference<decltype(*begin)>;
  return SimpleView<T>{std::forward<Iterator>(begin), std::forward<Iterator>(end)};
}

template <typename Container>
auto make_simple_view(Container&& container){
  using T = std::remove_reference<decltype(*container.begin())>;
  return SimpleView<T>{container};
}

} // end namespace MonteRay

#endif
