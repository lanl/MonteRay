#ifndef NEW_MONTERAY_MATERIALPROPERTIES_HH_
#define NEW_MONTERAY_MATERIALPROPERTIES_HH_

#include "MonteRayAssert.hh"
#include "MonteRay_MaterialProperties_FlatLayout.hh"
#include "SimpleVector.hh"

namespace MonteRay{

template <template <typename... T> class Container>
class MaterialProperties_Data_t {
  private: 

  using Offset_t = size_t;
  using Material_Index_t = MonteRay_CellProperties::Material_Index_t;
  using Temperature_t = MonteRay_CellProperties::Temperature_t;
  using MatID_t = MonteRay_CellProperties::MatID_t;
  using Density_t = MonteRay_CellProperties::Density_t;

  Container<Offset_t> offset_;
  Container<MatID_t> IDs_;
  Container<Density_t> densities_;

  constexpr auto checkInputsAndGetIndex( unsigned cellNum, unsigned matNum ) const { 
    MONTERAY_ASSERT_MSG( cellNum < this->numCells(), "MonteRay::MaterialProperties_Data::get...(cellNum,matNum)  -- requested cell number exceeds number of allocated cells!" );
    MONTERAY_ASSERT_MSG( matNum < numMats(cellNum), "MonteRay::MaterialProperties_Data::get...(cellNum,matNum)  -- requested material exceeds number of materials in the cell!" );
    auto index = offset_[cellNum] + matNum;
    MONTERAY_ASSERT_MSG( index < this->numMaterialComponents(), "MonteRay::MaterialProperties_Data::get...(cellNum,matNum)  -- requested material index exceeds number of total number of materials.!" );
    return index;
  }

  MaterialProperties_Data_t(Container<Offset_t>&& offset, Container<MatID_t>&& ID, Container<Density_t>&& density):
    offset_(offset), IDs_(ID), densities_(density) 
  {} 

  public:

  constexpr auto numCells() const { return offset_.size() - 1; }
  constexpr auto getNumCells() const { return this->numCells(); }
  constexpr auto numMats( unsigned i ) const { return offset_[i + 1] - offset_[i]; }
  constexpr auto getNumMats( unsigned i ) const { return this->numMats(i); }
  constexpr auto numMaterialComponents() const { return densities_.size(); }
  constexpr auto getDensity( unsigned cellNum, unsigned matNum ) const { 
    return densities_[ checkInputsAndGetIndex(cellNum, matNum) ]; 
  }
  constexpr auto getMatID( unsigned cellNum, unsigned matNum ) {
    return IDs_[ checkInputsAndGetIndex(cellNum, matNum) ]; 
  }
  // TPB TODO: Sace these functions and see they are useful later
  /* constexpr const auto cellMaterialIDs const (unsigned cellNum) { */
  /*   return make_simple_view( &IDs_[offset_[cellNum]], &IDs_[offset_[cellNum + 1]] ); */
  /* } */
  /* constexpr auto const cellMaterialDensities const (unsigned cellNum) { */
  /*   return make_simple_view( &static_cast<const Density_t>(densities_[offset_[cellNum]]), */ 
  /*       &static_cast<const Density_t>(densities_[offset_[cellNum + 1]]) ); */
  /* } */

  class Builder {
    private:
    struct Cell{
      std::vector<MatID_t> ids;
      std::vector<Density_t> densities;
    };
    std::vector<Cell> b_cells_;

    public:
    /* Builder (int numCells): b_cells_{numCells} {} */
    Builder() = default;
    
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

    MaterialProperties_Data_t build(){
      Container<Offset_t> offset;
      Container<MatID_t> IDs;
      Container<Density_t> densities;

      offset.emplace_back(0);
      for (auto& cell : b_cells_){
        offset.emplace_back(offset.back() + cell.ids.size());
        IDs.insert(IDs.end(), cell.ids.begin(), cell.ids.end());
        densities.insert(densities.end(), cell.densities.begin(), cell.densities.end());
      }

      return {std::move(offset), std::move(IDs), std::move(densities)};
    }
  };
};

using MaterialProperties_Data = MaterialProperties_Data_t<SimpleVector>;

} // end namespace MonteRay

#endif // NEW_MONTERAY_MATERIALPROPERTIES_HH_
