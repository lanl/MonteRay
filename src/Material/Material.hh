#ifndef NEWMONTERAYMATERIAL_HH_
#define NEWMONTERAYMATERIAL_HH_

#include <fstream>
#include <numeric>

#include "SimpleVector.hh"
#include "MonteRayTypes.hh"
#include "MonteRayConstants.hh"
#include "MonteRay_binaryIO.hh"

namespace MonteRay{

// inherit from tuple to get the constructor.  Make member functions to improve readability via hiding std::get<>();
template <class CrossSection>
struct CrossSectionAndFraction: public std::tuple<CrossSection, gpuFloatType_t>
{
  using std::tuple<CrossSection, gpuFloatType_t>::tuple;
  constexpr auto& xs() noexcept { return std::get<0>(*this); }
  constexpr auto& fraction() noexcept { return std::get<1>(*this); }
  constexpr const auto& xs() const noexcept { return std::get<0>(*this); }
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
  constexpr auto xs(unsigned i) const noexcept { return xsAndFracs_[i].xs(); }
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

  class Builder
  {
    gpuFloatType_t b_atomicWeight_;
    SimpleVector<CrossSectionAndFraction<CrossSection>> b_xsAndFracs_;

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

    void addIsotope(gpuFloatType_t frac, const CrossSection& xs){
      b_xsAndFracs_.emplace_back(xs, frac);
    }

    void addIsotope(gpuFloatType_t frac, CrossSection&& xs){
      b_xsAndFracs_.emplace_back(std::move(xs), frac);
    }

    auto build(){
      normalizeFractions();
      return Material(std::move(b_xsAndFracs_), calcAtomicWeight());
    }
  };


void write(std::ostream& outf) const{
  binaryIO::write(outf, this->numIsotopes());
  binaryIO::write(outf, this->atomicWeight() );
  for (auto&& xsAndFrac : xsAndFracs_){
    binaryIO::write(outf, xsAndFrac.fraction());
  }
  for (auto&& xsAndFrac : xsAndFracs_){
    xsAndFrac.xs().write(outf);
  }
}

void writeToFile( const std::string& filename ) const {
  std::ofstream outfile;
  outfile.open( filename.c_str(), std::ios::binary | std::ios::out);
  if( ! outfile.is_open() ) {
    fprintf(stderr, "Material::writeToFile -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
    throw std::runtime_error("Material::writeToFile -- Failure to open file" );
  }
  assert( outfile.good() );
  outfile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
  write( outfile );
  outfile.close();
}


void read(std::istream& infile) {
  size_t nIsotopes;
  binaryIO::read(infile, nIsotopes);
  binaryIO::read(infile, atomicWeight_);
  std::vector<gpuFloatType_t> fractions(nIsotopes);
  for( auto& frac : fractions ) binaryIO::read(infile, frac);
  std::vector<CrossSection> crossSections(nIsotopes);
  for( auto& frac : fractions){
    CrossSection xs;
    xs.read( infile );
    xsAndFracs_.emplace_back(std::move(xs), frac);
  }
}

void readFromFile( const std::string& filename) {
  std::ifstream infile;
  if( infile.is_open() ) {
      infile.close();
  }
  infile.open( filename.c_str(), std::ios::binary | std::ios::in);

  if( ! infile.is_open() ) {
      fprintf(stderr, "Error:  Material::readFromFile -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
      throw std::runtime_error("Material::readFromFile -- Failure to open file" );
  }
  assert( infile.good() );
  infile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
  read(infile);
  infile.close();
}

};

} // end namespace MonteRay

#endif
