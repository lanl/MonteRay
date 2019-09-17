#include <UnitTest++.h>

#include <sstream>
#include <memory>

#include "MaterialList.hh"
#include "SimpleVector.hh"

struct CrossSectionList { };

struct Material{
  int testVal = 0;
  void write(std::ostream& stream) const {
    MonteRay::binaryIO::write(stream, testVal);
  }

  template <typename XSList>
  struct Builder{
    int b_testVal = 0;
    Builder(XSList) {}
    void read(std::istream& stream){
      MonteRay::binaryIO::read(stream, b_testVal);
    }
    Material build() { 
      return Material{b_testVal};
    }
  };
};

using MaterialList = MonteRay::MaterialList_t<Material, MonteRay::SimpleVector>;

SUITE( MaterialList_tester ) {

  TEST( build_MaterialList ){
    auto mlb = MaterialList::Builder{1, Material{9}};
    mlb.addMaterial(2, Material{99});
    mlb.addMaterial(5, Material{999});
    auto materialList = mlb.build();

    CHECK_EQUAL(3, materialList.size());

    CHECK_EQUAL(1, materialList.id(0));
    CHECK_EQUAL(2, materialList.id(1));
    CHECK_EQUAL(5, materialList.id(2));

    CHECK_EQUAL(9,   materialList.material(0).testVal);
    CHECK_EQUAL(99,  materialList.material(1).testVal);
    CHECK_EQUAL(999, materialList.material(2).testVal);

  }

  TEST( read_write_MaterialList ){
    auto mlb = MaterialList::Builder{1, Material{9}};
    mlb.addMaterial(2, Material{99});
    mlb.addMaterial(5, Material{999});
    auto materialList = mlb.build();

    std::stringstream stream;
    materialList.write(stream);
    MaterialList::Builder another_mlb;
    another_mlb.read(stream, CrossSectionList{});
    auto another_materialList = another_mlb.build();

    CHECK_EQUAL(3, another_materialList.size());

    CHECK_EQUAL(1, another_materialList.id(0));
    CHECK_EQUAL(2, another_materialList.id(1));
    CHECK_EQUAL(5, another_materialList.id(2));

    CHECK_EQUAL(9,   another_materialList.material(0).testVal);
    CHECK_EQUAL(99,  another_materialList.material(1).testVal);
    CHECK_EQUAL(999, another_materialList.material(2).testVal);
  }

#ifdef __CUDACC__

  __global__ void func (int* val, MaterialList* mlp) {
    *val = mlp->size();
    *val += mlp->id(0);
    *val *= mlp->material(0).testVal;
  };

  TEST( access_MaterialList_on_GPU )
  {
    auto mlb = MaterialList::Builder{1, Material{9}};
    auto mlp = std::make_unique<MaterialList>(mlb.build());
    auto ml = mlb.build();
    int* val;
    cudaMallocManaged(&val, sizeof(int));

    func<<<1, 1>>>(val, mlp.get());
    cudaDeviceSynchronize();
    CHECK_EQUAL(*val, 18);
  }
#endif

  TEST( material_id_to_index ) {
    MonteRay::SimpleVector<unsigned> vec {1, 2};
    auto mlb = MaterialList::Builder{1, Material{9}};
    mlb.addMaterial(2, Material{99});
    mlb.addMaterial(5, Material{999});
    auto materialList = mlb.build();

    CHECK_EQUAL(2, materialList.materialIDtoIndex(5));
    CHECK_EQUAL(1, materialList.materialIDtoIndex(2));
    CHECK_THROW(materialList.materialIDtoIndex(10), std::runtime_error);
  }

}
