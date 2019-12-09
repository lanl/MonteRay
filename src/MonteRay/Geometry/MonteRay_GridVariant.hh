#ifndef MONTERAY_GRIDVARIANT_HH_
#define MONTERAY_GRIDVARIANT_HH_

#include "ManagedAllocator.hh"
#include "MonteRay_TransportMeshTypeEnum.hh"

#include "MonteRay_CartesianGrid.t.hh"
#include "MonteRay_CylindricalGrid.t.hh"
/* #include "MonteRay_SphericalGrid.t.hh" */

/* #include <mpark/variant.hpp> */


namespace MonteRay {

class GridVariant : public Managed {

  private:
  TransportMeshType coordinateSystem;
  MonteRay_CartesianGrid cartGrid;
  MonteRay_CylindricalGrid cylGrid;
  /* MonteRay_SphericalGrid sphereGrid; */

  public:
  GridVariant() = default;

  GridVariant(MonteRay_CartesianGrid grid) : coordinateSystem(TransportMeshType::Cartesian), cartGrid(std::move(grid)) {}
  GridVariant(MonteRay_CylindricalGrid grid) : coordinateSystem(TransportMeshType::Cylindrical), cylGrid(std::move(grid)) {}

  template <typename... Args>
  GridVariant(TransportMeshType meshType, Args&&... args): coordinateSystem(meshType) {
    switch (meshType) {
      case TransportMeshType::Cartesian:
        cartGrid = MonteRay_CartesianGrid(3, std::forward<Args>(args)...);
        break;
      case TransportMeshType::Cylindrical:
        cylGrid = MonteRay_CylindricalGrid(2, std::forward<Args>(args)...);
        break;
      case TransportMeshType::Spherical:
        throw std::runtime_error("Error: SphericalGrid is not supported yet");
        /* sphereGrid = MonteRay_SphericalGrid(1, std::forward<Args>(args)...); */
        /* break; */
    }
  }

  CUDA_CALLABLE_MEMBER auto getCoordinateSystem() const { return coordinateSystem; }

#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
  template<typename Func>
  CUDA_CALLABLE_MEMBER auto visit(Func&& func) const {
    switch (coordinateSystem) {
      case TransportMeshType::Cartesian:
        return func(cartGrid);
      case TransportMeshType::Cylindrical:
        return func(cylGrid);
      default:
        printf("Error: Unkown grid type in GridVariant::Visit");
        return func(cartGrid);
    }
  }

};

} // end namespace MonteRay

#endif
