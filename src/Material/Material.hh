#ifndef NEWMONTERAYMATERIAL_HH_
#define NEWMONTERAYMATERIAL_HH_

#include <numeric>
#include <functional>
#include <string>

#include "SimpleVector.hh"
#include "MonteRayTypes.hh"
#include "MonteRayConstants.hh"
#include "MonteRay_binaryIO.hh"

namespace MonteRay{

// inherit from tuple to get the constructor.  Make member functions to improve readability via hiding std::get<>();
template <class CrossSection>
struct CrossSectionAndFraction: public std::tuple<const CrossSection*, gpuFloatType_t>
{
  using std::tuple<const CrossSection*, gpuFloatType_t>::tuple;
  constexpr auto& xs() noexcept { return *std::get<0>(*this); }
  constexpr auto& fraction() noexcept { return std::get<1>(*this); }
  constexpr const auto& xs() const noexcept { return *std::get<0>(*this); }
  constexpr const auto& fraction() const noexcept { return std::get<1>(*this); }
};

template <typename CrossSection>
class Material{
private:
  gpuFloatType_t atomicWeight_;
  SimpleVector<CrossSectionAndFraction<CrossSection>> xsAndFracs_;

public:
  Material() = default;
  Material(SimpleVector<CrossSectionAndFraction<CrossSection>>&& xsAndFracs, gpuFloatType_t atomicWeight):
    xsAndFracs_(xsAndFracs), atomicWeight_(atomicWeight)
  { }

  constexpr auto atomicWeight() const { return atomicWeight_; }
  constexpr const auto& xs(unsigned i) const noexcept { return xsAndFracs_[i].xs(); }
  constexpr auto fraction(unsigned i) const noexcept { return xsAndFracs_[i].fraction(); }
  constexpr auto numIsotopes() const noexcept { return xsAndFracs_.size(); }

  constexpr auto getMicroTotalXS (gpuFloatType_t E) const noexcept {
    gpuFloatType_t total = static_cast<gpuFloatType_t>(0.0);
    for (auto&& xsAndFrac : xsAndFracs_){
      total += xsAndFrac.xs().getTotalXS(E)*xsAndFrac.fraction();
    }
    return total;
  }

  constexpr auto getTotalXS(gpuFloatType_t E, gpuFloatType_t density) const noexcept {
    return getMicroTotalXS(E) * density * AvogadroBarn / atomicWeight_;
  }

  template <typename Stream>
  void write(Stream&& outf) const{
    binaryIO::write(outf, this->numIsotopes());
    binaryIO::write(outf, this->atomicWeight() );
    for (auto&& xsAndFrac : xsAndFracs_){
      binaryIO::write(outf, xsAndFrac.xs().ZAID());
      binaryIO::write(outf, xsAndFrac.fraction());
    }
  }

  template <typename CrossSectionList>
  class Builder
  {
    gpuFloatType_t b_atomicWeight_;
    SimpleVector<CrossSectionAndFraction<CrossSection>> b_xsAndFracs_;
    std::reference_wrapper<const CrossSectionList> b_xsList_;

    auto calcAtomicWeight(){
      return gpu_neutron_molar_mass * std::accumulate(b_xsAndFracs_.begin(), b_xsAndFracs_.end(), 0.0, 
          [](auto&& sum, auto&& xsAndFrac){return sum + xsAndFrac.fraction()*xsAndFrac.xs().AWR();});
    }

    auto normalizeFractions(){
      auto invTotal = 1.0/std::accumulate(b_xsAndFracs_.begin(), b_xsAndFracs_.end(), 0.0, 
          [](auto&& sum, auto&& xsAndFrac){return sum + xsAndFrac.fraction();});
      std::for_each(b_xsAndFracs_.begin(), b_xsAndFracs_.end(), 
          [invTotal](auto&& xsAndFrac){ xsAndFrac.fraction() *= invTotal; });
    }

    public:

    Builder(const CrossSectionList& xsList): b_xsList_(std::ref(xsList)){ }

    void addIsotope(gpuFloatType_t frac, int zaid){
      const CrossSection* xsPtr = &(b_xsList_.get().getXSByZAID(zaid));
      if (xsPtr == nullptr){
        std::string message = "ZAID " + std::to_string(zaid) + " not present in CrossSectionList.";
        throw std::runtime_error(message);
      }
      b_xsAndFracs_.emplace_back(xsPtr, frac);
    }

    template <typename Stream>
    void read(Stream&& infile) {
      size_t nIsotopes;
      binaryIO::read(infile, nIsotopes);
      binaryIO::read(infile, b_atomicWeight_);
      b_xsAndFracs_.reserve(nIsotopes);
      for(int i = 0; i < nIsotopes; i++){
        int zaid;
        binaryIO::read(infile, zaid);
        gpuFloatType_t frac;
        binaryIO::read(infile, frac);
        this->addIsotope(frac, zaid);
      }
    }

    auto build(){
      normalizeFractions();
      return Material(std::move(b_xsAndFracs_), calcAtomicWeight());
    }
  };

  template <typename CrossSectionList>
  static auto make_builder(const CrossSectionList& xsList){
    return Material::Builder<CrossSectionList>(xsList);
  }

};

} // end namespace MonteRay

#endif
