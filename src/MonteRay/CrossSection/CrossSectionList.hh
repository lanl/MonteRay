#ifndef CROSSSECTIONLIST_HH_
#define CROSSSECTIONLIST_HH_

#define CROSSSECTION_LIST_VERSION static_cast<unsigned>(1)

#include "CrossSection.hh"
#include "SimpleVector.hh"

namespace MonteRay {

template<typename HashFunction = FasterHash>
class CrossSectionList_t : public Managed {
public:
  using CrossSection = CrossSection_t<HashFunction>;
private:
  SimpleVector<CrossSection> xsList_;

  CrossSectionList_t(SimpleVector<CrossSection>&& xsList): xsList_{std::move(xsList)} {}
public:
  constexpr size_t size() const { return xsList_.size(); }
  constexpr const CrossSection& xs(size_t i) const { return xsList_[i]; }
  constexpr const CrossSection& getXS(size_t i) const { return xs(i); }

  const CrossSection& getXSByZAID(int zaid) const {
    auto loc = std::find_if(xsList_.begin(), xsList_.end(), 
        [zaid](auto&& xs){ return xs.ZAID() == zaid; } );
    if (loc == xsList_.end()) {
      throw std::runtime_error("Attempted to access CrossSection with ZAID " + std::to_string(zaid) + 
          " in CrossSectionList but it doesn't exist.");
    }
    return *loc;
  }

  void write(std::ostream& stream) const {
    unsigned version = CROSSSECTION_LIST_VERSION;
    binaryIO::write(stream, version);
    binaryIO::write(stream, this->size());
    for (auto&& xs : xsList_ ) { xs.write(stream); }
  }

  class Builder{
    private:
    SimpleVector<CrossSection> b_xsList_;

    public:
    template <typename XS, std::enable_if_t< std::is_same<XS, CrossSection>::value, bool> = true>
    void add( XS&& xs ) {
      auto checkZaid = [&] (const CrossSection& list_xs) { return xs.ZAID() == list_xs.ZAID(); };
      auto xsLoc = std::find_if(b_xsList_.begin(), b_xsList_.end(), checkZaid);
      if (xsLoc != b_xsList_.end()){ return; }
      b_xsList_.emplace_back(std::forward<XS>(xs));
    }

    void read(std::istream& stream){
      unsigned version;
      binaryIO::read(stream, version);
      if (version != CROSSSECTION_LIST_VERSION) { 
        throw std::runtime_error("CrossSectionList file version " + std::to_string(version)  + 
            " is incompatible with expected version " + std::to_string(CROSSSECTION_LIST_VERSION));
      }
      decltype(b_xsList_.size()) numXS;
      binaryIO::read(stream, numXS);
      b_xsList_.reserve(numXS);
      for (size_t i = 0; i < numXS; i++){
        CrossSectionBuilder_t<HashFunction> xsBuilder;
        xsBuilder.read(stream);
        b_xsList_.emplace_back(xsBuilder.construct());
      }
    }

    CrossSectionList_t build() {
      return std::move(b_xsList_);
    }

  };
};

using CrossSectionList = CrossSectionList_t<>;

} /* namespace MonteRay */

#undef CROSSSECTION_LIST_VERSION
#endif /* CROSSSECTIONLIST_HH_ */
