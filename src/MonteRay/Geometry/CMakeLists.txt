target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/Geometry/MaterialProperties.hh"
  "${prefix}/MonteRay/Geometry/MonteRay_GridBins.hh"
  "${prefix}/MonteRay/Geometry/MonteRay_GridSystemInterface.t.hh"
  "${prefix}/MonteRay/Geometry/MonteRay_GridSystemInterface.hh"
  "${prefix}/MonteRay/Geometry/MonteRay_CartesianGrid.hh"
  "${prefix}/MonteRay/Geometry/MonteRay_CartesianGrid.t.hh"
  "${prefix}/MonteRay/Geometry/MonteRay_CylindricalGrid.hh"
  "${prefix}/MonteRay/Geometry/MonteRay_CylindricalGrid.t.hh"
#  "${prefix}/MonteRay/Geometry/MonteRay_SphericalGrid.hh"
#  "${prefix}/MonteRay/Geometry/MonteRay_SphericalGrid.t.hh"
  "${prefix}/MonteRay/Geometry/lnk3dnt.hh"
  "${prefix}/MonteRay/Geometry/MonteRay_TransportMeshTypeEnum.hh"
  "${prefix}/MonteRay/Geometry/MonteRay_GridVariant.hh"
  "${prefix}/MonteRay/Geometry/MonteRay_GridVariant.hh"
  "${prefix}/MonteRay/Geometry/MonteRay_SpatialGrid.hh"
PRIVATE
  "${prefix}/MonteRay/Geometry/MonteRay_GridBins.cu"
  "${prefix}/MonteRay/Geometry/MonteRay_GridSystemInterface.cu"
  "${prefix}/MonteRay/Geometry/MonteRay_CartesianGrid.cu"
  "${prefix}/MonteRay/Geometry/MonteRay_CylindricalGrid.cu"
  # "${prefix}/MonteRay/Geometry/MonteRay_SphericalGrid.cu"
  "${prefix}/MonteRay/Geometry/MonteRay_SpatialGrid.cu"
)

add_subdirectory(lnk3dnt)
target_include_directories(MonteRay PUBLIC
  "${prefix}/MonteRay/Geometry/lnk3dnt"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
endif()


