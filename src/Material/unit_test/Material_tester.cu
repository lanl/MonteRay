#include <UnitTest++.h>

#include <iostream>
#include <cstdio>

#include "Material.hh"
#include "Invokers.hh"

using namespace MonteRay;
struct CrossSection{
  gpuFloatType_t xs_;
  gpuFloatType_t AWR_;
  auto AWR() const { return AWR_; }
  auto getTotalXS(gpuFloatType_t) const { return xs_; }
  void write(std::ostream& outf) const { 
    binaryIO::write(outf, xs_);
    binaryIO::write(outf, AWR_);
  }
  void read(std::istream& infile) {
    binaryIO::read(infile, xs_);
    binaryIO::read(infile, AWR_);
  }
};

class MaterialFixture{
  public:
  using Material = Material<CrossSection>;
  Material mat;

  MaterialFixture(){
    auto mb = Material::Builder();
    CrossSection xs1{0.5, 1.0};
    CrossSection xs2{4.0, 2.0};
    CrossSection xs3{40.0, 3.0};
    mb.addIsotope(2.0, xs1);
    mb.addIsotope(3.0, xs2);
    mb.addIsotope(5.0, xs3);
    mat = mb.build();
  }
};


SUITE( Material_tester ) {

  constexpr double close = 1.0E-6;

  TEST_FIXTURE( MaterialFixture, builder ) {


    CHECK_CLOSE(mat.fraction(0), 0.2, close);
    CHECK_CLOSE(mat.fraction(1), 0.3, close);
    CHECK_CLOSE(mat.fraction(2), 0.5, close);

    gpuFloatType_t E = 1.0;
    CHECK_EQUAL(mat.xs(0).getTotalXS(E), 0.5);
    CHECK_EQUAL(mat.xs(1).getTotalXS(E), 4.0);
    CHECK_EQUAL(mat.xs(2).getTotalXS(E), 40.0);

    CHECK_EQUAL(mat.xs(0).AWR(), 1.0);
    CHECK_EQUAL(mat.xs(1).AWR(), 2.0);
    CHECK_EQUAL(mat.xs(2).AWR(), 3.0);

    CHECK_CLOSE(mat.atomicWeight(), neutron_molar_mass*(3.0*0.5 + 2.0*0.3 + 1.0*0.2), close);

    CHECK_EQUAL(mat.numIsotopes(), 3);

  }

  TEST_FIXTURE( MaterialFixture, TotalXS ) {
    gpuFloatType_t E = 1.0;
    gpuFloatType_t density = 2.0;
    CHECK_CLOSE(mat.getMicroTotalXS(E), 0.2*0.5 + 0.3*4.0 + 0.5*40, close);
    CHECK_CLOSE(mat.getTotalXS(E, density), mat.getMicroTotalXS(E) * density * AvogadroBarn / mat.atomicWeight(), close);

#ifdef __CUDACC__
    gpuFloatType_t* micro;
    cudaMallocManaged(&micro, sizeof(gpuFloatType_t));
    gpuFloatType_t* macro;
    cudaMallocManaged(&macro, sizeof(gpuFloatType_t));

    auto func = [=, mat = mat] __device__ () {
      *micro = mat.getMicroTotalXS(E);
      *macro = mat.getTotalXS(E, density);
    };

    d_invoker<<<1, 1>>>(func);
    cudaDeviceSynchronize();
    CHECK_CLOSE(*micro,  0.2*0.5 + 0.3*4.0 + 0.5*40, close);
    CHECK_CLOSE(*macro, mat.getMicroTotalXS(E) * density * AvogadroBarn / mat.atomicWeight(), close);
    cudaFree(micro);
    cudaFree(macro);
#endif

  }

  TEST_FIXTURE ( MaterialFixture, write_and_read ){
    std::string filename("MaterialTester.bin");
    mat.writeToFile(filename);
    Material newMat;
    newMat.readFromFile(filename);
    CHECK_EQUAL(newMat.atomicWeight(), mat.atomicWeight());
    CHECK_EQUAL(newMat.numIsotopes(), mat.numIsotopes());
    for (size_t i = 0; i < newMat.numIsotopes(); i++){
      CHECK_EQUAL(newMat.fraction(i), mat.fraction(i));
      CHECK_EQUAL(newMat.xs(i).xs_, mat.xs(i).xs_);
      CHECK_EQUAL(newMat.xs(i).AWR(), mat.xs(i).AWR());
    }
    std::remove(filename.c_str());
  }

}
