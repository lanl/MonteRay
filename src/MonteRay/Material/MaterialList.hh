#ifndef MONTERAY_MATERIALLIST_HH_
#define MONTERAY_MATERIALLIST_HH_

#include "ManagedAllocator.hh"
#include "MonteRay_binaryIO.hh"
#include <algorithm>

#define MATERIAL_LIST_VERSION 1
namespace MonteRay{

template <typename Material, template <typename... T> class Container = std::vector>
class MaterialList_t: public Managed {
  private:
  Container<Material> materials_;
  Container<unsigned> materialIDs_;

  MaterialList_t(Container<Material>&& materials, Container<unsigned>&& materialIDs): 
    materials_(materials),
    materialIDs_(materialIDs) 
  {}

  public:
  constexpr auto size() const { return materials_.size(); }
  constexpr auto getNumberMaterials() const { return this->size(); }
  constexpr unsigned id(unsigned i) const { return materialIDs_[i]; }
  constexpr const Material& material(unsigned i) const { return materials_[i]; }
  constexpr Material& material(unsigned i) { return materials_[i]; }

  void write(std::ostream& outfile) const {
    unsigned version = MATERIAL_LIST_VERSION;
    binaryIO::write(outfile, version);
    binaryIO::write(outfile, this->size());
    for(auto&& mat : materials_){
      mat.write(outfile);
    }
    for(auto&& id : materialIDs_){
      binaryIO::write(outfile, id);
    }
  }

  auto materialIDtoIndex(unsigned id) const {
    auto iter = std::find(materialIDs_.begin(), materialIDs_.end(), id);
    if (iter != materialIDs_.end()) {
      return std::distance(materialIDs_.begin(), iter);
    } else {
      throw std::runtime_error("Material ID " + std::to_string(id) + " not found in MaterialList_t.");
    }
  }

  class Builder{
    private:
    Container<Material> b_materials_;
    Container<unsigned> b_materialIDs_;

    template<typename MaterialT>
    using is_material_e = std::enable_if_t< 
      std::is_constructible<Material, MaterialT>::value or
      std::is_same<Material, MaterialT>::value,
      bool >;

    public:
    Builder() = default;

    template <typename MaterialT, is_material_e<MaterialT> = true>
    Builder(unsigned id, MaterialT&& mat) {
      b_materials_.emplace_back(std::forward<MaterialT>(mat));
      b_materialIDs_.emplace_back(id);
    }

    template <typename MaterialT, is_material_e<MaterialT> = true>
    Builder& addMaterial(unsigned id, MaterialT&& mat){
      b_materials_.emplace_back(std::forward<MaterialT>(mat));
      b_materialIDs_.emplace_back(id);
      return *this;
    }

    template <typename CrossSectionList>
    void read(std::istream& stream, const CrossSectionList& xsList){
      unsigned version;
      binaryIO::read(stream, version);
      if (version != MATERIAL_LIST_VERSION){
        throw std::runtime_error("MaterialList_t binary IO: file version " + std::to_string(version) + 
            " incompatible with required version " + std::to_string(MATERIAL_LIST_VERSION));
      }
      decltype(b_materials_.size()) numMaterials;
      binaryIO::read(stream, numMaterials);
      b_materials_.reserve(numMaterials);
      b_materialIDs_.resize(numMaterials);
      for (size_t i = 0; i < numMaterials; i++){
        typename Material::template Builder<CrossSectionList> mb(xsList);
        mb.read(stream);
        b_materials_.emplace_back(mb.build());
      }
      for (auto& id : b_materialIDs_){
        binaryIO::read(stream, id);
      }
    }

    MaterialList_t build() { 
      if (b_materials_.size() == 0) {
        throw std::runtime_error("Building empty MaterialList_t");
      }
      return MaterialList_t{std::move(b_materials_), std::move(b_materialIDs_)}; }
  };
};

#undef MATERIAL_LIST_VERSION
} // end namespace MonteRay

#include "Material.hh"
#include "SimpleVector.hh"
namespace MonteRay{
  using MaterialList = MaterialList_t<Material, SimpleVector>;
}

#endif
