#ifndef NEW_MONTERAY_MATERIALPROPERTIES_HH_
#define NEW_MONTERAY_MATERIALPROPERTIES_HH_

#include "MonteRayAssert.hh"
#include "MonteRay_binaryIO.hh"
#include "ManagedAllocator.hh"
#include <iostream>
#include <algorithm>
#define MATERIAL_PROPERTIES_VERSION 1

namespace MonteRay{

template <template <typename... T> class Container>
class MaterialProperties_t: public Managed {
  public:
    using Offset_t = size_t;
    using Material_Index_t = unsigned;
    using MatID_t = short int;
    using Density_t = gpuFloatType_t;
    using Cell_Index_t = int;
  private: 


  Container<Offset_t> offset_;
  Container<MatID_t> IDs_;
  Container<Density_t> densities_;

  constexpr auto checkInputsAndGetIndex( unsigned cellNum, unsigned matNum ) const { 
    MONTERAY_ASSERT_MSG( cellNum < this->numCells(), "MonteRay::MaterialProperties::get...(cellNum,matNum)  -- requested cell number exceeds number of allocated cells!" );
    MONTERAY_ASSERT_MSG( matNum < numMats(cellNum), "MonteRay::MaterialProperties::get...(cellNum,matNum)  -- requested material exceeds number of materials in the cell!" );
    auto index = offset_[cellNum] + matNum;
    MONTERAY_ASSERT_MSG( index < this->numMaterialComponents(), "MonteRay::MaterialProperties::get...(cellNum,matNum)  -- requested material index exceeds number of total number of materials.!" );
    return index;
  }

  MaterialProperties_t(Container<Offset_t>&& offset, Container<MatID_t>&& ID, Container<Density_t>&& density):
    offset_(offset), IDs_(ID), densities_(density) 
  {} 

  public:

  template <typename OtherMaterialProperties>
  MaterialProperties_t(const OtherMaterialProperties& other){
		auto nCells = other.size();
    offset_.assign(other.getOffsetData(), other.getOffsetData() + nCells + 1);
		auto nMaterialComponents = other.numMatSpecs();
    IDs_.assign(other.getMaterialIDData(), other.getMaterialIDData() + nMaterialComponents);
    densities_.assign(other.getMaterialDensityData(), other.getMaterialDensityData() + nMaterialComponents);
  }

  constexpr auto numCells() const { return offset_.size() - 1; }
  constexpr auto getNumCells() const { return this->numCells(); }
  constexpr auto numMats( unsigned i ) const { return offset_[i + 1] - offset_[i]; }
  constexpr auto getNumMats( unsigned i ) const { return this->numMats(i); }
  constexpr auto numMaterials( unsigned i ) const { return this->numMats(i); }
  constexpr auto numMaterialComponents() const { return densities_.size(); }
  constexpr auto getDensity( unsigned cellNum, unsigned matNum ) const { return densities_[ checkInputsAndGetIndex(cellNum, matNum) ]; }
  constexpr auto getMaterialDensity( unsigned cellNum, unsigned matNum ) const { return this->getDensity(cellNum, matNum); }
  constexpr auto getMaterialID( unsigned cellNum, unsigned matNum ) const { return IDs_[ checkInputsAndGetIndex(cellNum, matNum) ]; }
  constexpr auto getMatID( unsigned cellNum, unsigned matNum ) const { return this->getMaterialID(cellNum, matNum); }
  // TPB TODO: Save these functions and see they are useful later
  /* constexpr const auto cellMaterialIDs const (unsigned cellNum) { */
  /*   return make_simple_view( &IDs_[offset_[cellNum]], &IDs_[offset_[cellNum + 1]] ); */
  /* } */
  /* constexpr auto const cellMaterialDensities const (unsigned cellNum) { */
  /*   return make_simple_view( &static_cast<const Density_t>(densities_[offset_[cellNum]]), */ 
  /*       &static_cast<const Density_t>(densities_[offset_[cellNum + 1]]) ); */
  /* } */

  auto getPtr() { std::cout << " Warning: getPtr() is deprecated."; return this; }
  const auto getPtr() const { std::cout << " Warning: getPtr() is deprecated."; return this; }

  // TODO: implement write/read
  void writeToFile(const std::string&) const {

  }

  void write(std::ostream& stream) const {
    unsigned version = MATERIAL_PROPERTIES_VERSION;
    binaryIO::write(stream, version);

    binaryIO::write(stream, offset_.size());
    for (auto& val : offset_){
      binaryIO::write(stream, val);
    }

    binaryIO::write(stream, IDs_.size());
    for (auto& val : IDs_){
      binaryIO::write(stream, val);
    }
    for (auto& val : densities_){
      binaryIO::write(stream, val);
    }
  }

  template <typename MaterialList>
  void renumberMaterialIDs(const MaterialList& matList) {
    for (auto& id : IDs_){
      id = matList.materialIDtoIndex(id);
    }
  }



  class Builder {
    public:
    struct Cell{
      std::vector<MatID_t> ids;
      std::vector<Density_t> densities;
      auto getMaterialID(size_t i){ return ids[i]; }
      auto getMaterialDensity(size_t i){ return densities[i]; }
      auto getNumMaterials() { return ids.size(); }
      auto& add(MatID_t matID, Density_t density){
        ids.push_back(matID);
        densities.push_back(density);
        return *this;
      }
    };
    private:
    std::vector<Cell> b_cells_;
    bool memoryReductionDisabled = false;

    size_t getEqualNumMatMemorySize(Cell_Index_t nCells, size_t nMats ) const {
        size_t total = 0;
        total += sizeof(MatID_t)*nMats*nCells;
        total += sizeof(Density_t)*nMats*nCells;
        return total;
    }

    size_t getNonEqualNumMatMemorySize(Cell_Index_t nCells, size_t nMatComponents) const {
        size_t total = 0;
        total += sizeof(Offset_t)*(nCells+1);
        total += sizeof(MatID_t)*nMatComponents;
        total += sizeof(Density_t)*nMatComponents;
        return total;
    }

    public:
    /* Builder (int numCells): b_cells_{numCells} {} */
    Builder() = default;


    template<typename T>
    Builder(T&& obj){
      setMaterialDescription(std::forward<T>(obj));
    }

    void readFromFile(const std::string&) const {}

    MaterialProperties_t read(std::istream& stream){
      unsigned version;
      binaryIO::read(stream, version);
      if (version != MATERIAL_PROPERTIES_VERSION){
        throw std::runtime_error("MaterialProperties binary IO: file version " + std::to_string(version) + 
            " incompatible with required version " + std::to_string(MATERIAL_PROPERTIES_VERSION));
      }

      using size_type = decltype(IDs_.size());
      size_type nOffsets;
      binaryIO::read(stream, nOffsets);
      Container<Offset_t> offset(nOffsets);
      for (auto& val : offset){
        binaryIO::read(stream, val);
      }

      size_type nComponents;
      binaryIO::read(stream, nComponents);
      Container<MatID_t> IDs(nComponents);
      Container<Density_t> densities(nComponents);
      for (auto& val : IDs){
        binaryIO::read(stream, val);
      }
      for (auto& val : densities){
        binaryIO::read(stream, val);
      }
      return {std::move(offset), std::move(IDs), std::move(densities)};
    }

    void disableMemoryReduction() { memoryReductionDisabled = true; }
    
    template <typename OtherCell>
    Builder& addCell(OtherCell&& otherCell){
      Cell cell;
      auto N = otherCell.getNumMaterials();
      for (decltype(N) i = 0; i < N; i++){
        cell.ids.push_back(otherCell.getMaterialID(i));
        cell.densities.push_back(otherCell.getMaterialDensity(i));
      }
      b_cells_.emplace_back(std::move(cell));
      return *this;
    }

    Builder& addCellMaterial(int cellID, MatID_t matID, Density_t density){
      if (cellID > b_cells_.size()) throw std::runtime_error("Error while building MaterialProperties: " 
          " Trying to set property of CellID: " + std::to_string(cellID) + 
          " which is beyond number of cells: " + std::to_string(b_cells_.size()));
      auto& matIDs = b_cells_[cellID].ids;
      auto& densities = b_cells_[cellID].densities;
      auto found = std::find(matIDs.begin(), matIDs.end(), matID);
      if (found != matIDs.end()){
        throw std::runtime_error("Unable to add material to cell.  Material " + std::to_string(matID) + 
            " already exists in cell " + std::to_string(cellID));
      } else {
        matIDs.push_back(matID);
        densities.push_back(density);
      }
      return *this;
    }

    ///Initializes the material description using vectors of matIDs and density.
    // TPB: this should be in client code.
    template<typename MaterialIDType, typename DensityType>
    void initializeMaterialDescription( const std::vector<MaterialIDType>& matIDs, 
        const std::vector<DensityType>& densities, const std::size_t nCells) {
      if( matIDs.size() != densities.size() ) {
          std::stringstream msg;
          msg << "Material ID vector size is not the same as density vector size!\n";
          msg << "Material ID vector size = " << matIDs.size() << ", density vector size = " << densities.size() << "\n";
          msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MaterialProperties::Builder::initializeMaterialDescription" << "\n\n";
          throw std::runtime_error( msg.str() );
      }

      MaterialIDType maxMatID = std::numeric_limits<MatID_t>::max();

      MatID_t NULL_MATERIAL = std::numeric_limits<MatID_t>::min();

      // get the max number of non-zero density materials in a cell
      MatID_t maxNumComponents = 0;
      size_t nComponents = 0;
      for( size_t n=0; n<nCells; ++n ) {
        unsigned numCellMats = 0;
        for( size_t index=n; index < matIDs.size(); index += nCells ) {
          if( matIDs[index] <= NULL_MATERIAL || matIDs[index] > maxMatID ) {
              std::stringstream msg;
              msg << "Material ID exceeds MatID_t range!\n";
              msg << "Material ID = " << matIDs[index] << ", upper range limit = " 
                <<  maxMatID << ", lower range limit = " << NULL_MATERIAL << "\n";
              msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MaterialProperties::Builder::initializeMaterialDescription" << "\n\n";
              std::cout << "MonteRay Error: " << msg.str();
              throw std::runtime_error( msg.str() );
          }

          if( densities[index] > 1.0e-10 ) {
              ++numCellMats;
              ++nComponents;
          }
        }
        if(numCellMats > maxNumComponents ) {
            maxNumComponents = numCellMats;
        }
      }

      size_t memorySizeForEqualNumMats = getEqualNumMatMemorySize( nCells, maxNumComponents );
      size_t memorySizeForNonEqualNumMats = getNonEqualNumMatMemorySize( nCells, nComponents );
      std::cout << memorySizeForEqualNumMats << "  " << memorySizeForNonEqualNumMats << std::endl;
      bool singleNumComponents = ( memorySizeForEqualNumMats < memorySizeForNonEqualNumMats && !memoryReductionDisabled );

      Cell cell;
      cell.ids.reserve(nCells * maxNumComponents);
      cell.densities.reserve(nCells * maxNumComponents);
      for( size_t n=0; n<nCells; ++n ) {
        for( size_t index=n; index < matIDs.size(); index += nCells ) {
          int ID = matIDs[ index ];
          DensityType matDensity = densities[ index ];
          if( matDensity > 1.0e-10 ) {
            cell.ids.push_back(ID);
            cell.densities.push_back(matDensity);
          }
        }

        if( singleNumComponents ) { // fillup blank section of cell materials
          for( unsigned i=cell.ids.size(); i<maxNumComponents; ++i ){
            cell.ids.push_back(NULL_MATERIAL);
          }
          for( unsigned i=cell.densities.size(); i<maxNumComponents; ++i ){
            cell.densities.push_back(0.0);
          }
        }

        this->addCell( cell );
        cell.ids.clear();
        cell.densities.clear();
      }
    }

    ///Set material description from another object, like lnk3dnt.
    template<typename objType>
    void setMaterialDescription(const objType& obj){
      size_t nCells = obj.NCells(0) * obj.NCells(1) * obj.NCells(2);

      size_t NMaxMaterialsPerCell = obj.MaxMaterialsPerCell();
      std::vector< double > density( nCells * NMaxMaterialsPerCell );
      std::vector< int >   material( nCells * NMaxMaterialsPerCell );

      obj.fillDensityArray ( density.data()  );
      obj.fillMaterialArray( material.data() );

      initializeMaterialDescription( material, density, nCells );
    }

    ///Set material description from another object, like lnk3dnt with view.
    template<typename objType, typename iter>
    void setMaterialDescription( objType& obj, const iter& view ){
      std::vector< double > density;
      std::vector< int >    material;
      obj.extract(material, density, view );
      initializeMaterialDescription( material, density, view.size() );
    }

    template <typename MaterialList>
    void renumberMaterialIDs(const MaterialList& matList) {
      for (auto& cell: b_cells_){
        for (auto& id : cell.ids){
          id = matList.materialIDtoIndex(id);
        }
      }
    }

    MaterialProperties_t build(){
      Container<Offset_t> offset;
      Container<MatID_t> IDs;
      Container<Density_t> densities;

      offset.reserve(b_cells_.size() + 1);
      offset.emplace_back(0);
      for (auto& cell : b_cells_){
        offset.emplace_back(offset.back() + cell.ids.size());
      }
      IDs.reserve(offset.back());
      densities.reserve(offset.back());
      for (auto& cell : b_cells_){
        IDs.insert(IDs.end(), cell.ids.begin(), cell.ids.end());
        densities.insert(densities.end(), cell.densities.begin(), cell.densities.end());
      }
      return {std::move(offset), std::move(IDs), std::move(densities)};
    }
  };
};

} // end namespace MonteRay

#include "SimpleVector.hh"
namespace MonteRay{
  using MaterialProperties = MaterialProperties_t<SimpleVector>;
}

#endif // NEW_MONTERAY_MATERIALPROPERTIES_HH_
