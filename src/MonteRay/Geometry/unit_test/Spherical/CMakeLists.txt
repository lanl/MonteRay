add_executable( monteray.geometry.spherical.test )
setCudaFilesToCppFiles()
target_sources( monteray.geometry.spherical.test  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../main.cc
  ${CMAKE_CURRENT_LIST_DIR}/../MonteRay_SpatialGrid_GPU_helper.hh
  ${CMAKE_CURRENT_LIST_DIR}/../MonteRay_SpatialGrid_GPU_helper.cu
  ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SphericalGrid_GPU_test.cu
  ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SphericalGrid_crossingDistance_GPU_tests.cu
  ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SphericalGrid_crossingDistance_tests.cu
  ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SpatialGrid_Spherical_test.cu
  ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SphericalGrid_rayTrace_tests.cu
  ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SphericalGrid_test.cu
  ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SphericalGrid_rayTrace_GPU_tests.cu
  ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SpatialGrid_Spherical_GPU_tester.cu
)

target_link_libraries( monteray.geometry.spherical.test PRIVATE MonteRay::testing )
add_test(NAME monteray.geometry.spherical.test COMMAND monteray.geometry.spherical.test) 
create_lnk3dnt_symlink(monteray.geometry.spherical.test)
