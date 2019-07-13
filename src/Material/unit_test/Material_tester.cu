#include <UnitTest++.h>

#include <iostream>
#include <cstdio>

#include "Material.hh"
#include "GPUUtilityFunctions.hh"

using namespace MonteRay;

struct CrossSection{
  int zaid;
  auto ZAID() const {return zaid;}
  gpuFloatType_t AWR() const {return 1.0;}
  gpuFloatType_t getTotalXS(gpuFloatType_t) const {return 1.0;}
};

struct CrossSectionList{
  std::vector<CrossSection> xs_vec;

  const CrossSection* getXSByZAID(int ZAID) const {
    auto loc = std::find_if(xs_vec.begin(), xs_vec.end(), 
        [ZAID](auto&& xs){ return xs.ZAID() == ZAID; } );
    const CrossSection* retval = (loc != xs_vec.end()) ?  &(*loc) : nullptr;
    return retval;
  }
};


class MaterialFixture{
  public:
  using Material = Material<CrossSection>;
  CrossSectionList xsList;
  Material mat;

  MaterialFixture(){
    using XS = CrossSection;
    xsList = CrossSectionList{{XS{1001}, XS{2004}, XS{6012}}};
    auto mb = Material::make_builder(xsList);
    mb.addIsotope(2.0, 1001);
    mb.addIsotope(3.0, 2004);
    mb.addIsotope(5.0, 6012);
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
    CHECK_EQUAL(1001, mat.xs(0).ZAID());
    CHECK_EQUAL(2004, mat.xs(1).ZAID());
    CHECK_EQUAL(6012, mat.xs(2).ZAID());

    CHECK_CLOSE(mat.atomicWeight(), neutron_molar_mass, close);

    CHECK_EQUAL(mat.numIsotopes(), 3);

  }

  /* TEST_FIXTURE( MaterialFixture, TotalXS ) { */
  /*   gpuFloatType_t E = 1.0; */
  /*   gpuFloatType_t density = 2.0; */
  /*   CHECK_CLOSE(mat.getMicroTotalXS(E), 1.0, close); */
  /*   CHECK_CLOSE(mat.getTotalXS(E, density), mat.getMicroTotalXS(E) * density * AvogadroBarn / mat.atomicWeight(), close); */

/* #ifdef __CUDACC__ */
  /*   int* zaid; */
  /*   gpuFloatType_t* micro; */
  /*   cudaMallocManaged(&micro, sizeof(gpuFloatType_t)); */
  /*   gpuFloatType_t* macro; */
  /*   cudaMallocManaged(&macro, sizeof(gpuFloatType_t)); */

  /*   auto func = [=, mat = mat] __device__ () { */
  /*     zaid = mat.xs(1).ZAID(); */
  /*     *micro = mat.getMicroTotalXS(E); */
  /*     *macro = mat.getTotalXS(E, density); */
  /*   }; */

  /*   d_invoker<<<1, 1>>>(func); */
  /*   cudaDeviceSynchronize(); */
  /*   CHECK_EQUAL(*zaid, 2004); */
  /*   CHECK_CLOSE(*micro,  1.0, close); */
  /*   CHECK_CLOSE(*macro, mat.getMicroTotalXS(E) * density * AvogadroBarn / mat.atomicWeight(), close); */
  /*   cudaFree(micro); */
  /*   cudaFree(macro); */
/* #endif */

  /* } */

  /* TEST_FIXTURE ( MaterialFixture, write_and_read ){ */
  /*   std::string filename("MaterialTester.bin"); */
  /*   mat.writeToFile(filename); */
  /*   Material newMat; */
  /*   newMat.readFromFile(filename); */
  /*   CHECK_EQUAL(newMat.atomicWeight(), mat.atomicWeight()); */
  /*   CHECK_EQUAL(newMat.numIsotopes(), mat.numIsotopes()); */
  /*   for (size_t i = 0; i < newMat.numIsotopes(); i++){ */
  /*     CHECK_EQUAL(newMat.fraction(i), mat.fraction(i)); */
  /*     CHECK_EQUAL(newMat.xs(i).xs_, mat.xs(i).xs_); */
  /*     CHECK_EQUAL(newMat.xs(i).AWR(), mat.xs(i).AWR()); */
  /*   } */
  /*   std::remove(filename.c_str()); */
  /* } */

}
