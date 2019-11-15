add_executable( monteray.geometry.test )
setCudaFilesToCppFiles()
target_sources( monteray.geometry.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/genericGPU_test_helper.hh
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SpatialGrid_GPU_helper.hh
               ${CMAKE_CURRENT_LIST_DIR}/HashBins_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SphericalGrid_GPU_test.cu
               ${CMAKE_CURRENT_LIST_DIR}/GridBins_gpu_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SpatialGrid_GPU_helper.cu
               ${CMAKE_CURRENT_LIST_DIR}/MaterialProperties_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SphericalGrid_crossingDistance_GPU_tests.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SpatialGrid_GPU_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SphericalGrid_crossingDistance_tests.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SpatialGrid_Spherical_test.cu
               ${CMAKE_CURRENT_LIST_DIR}/GridBins_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/genericGPU_test_helper.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_GridBins_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SphericalGrid_rayTrace_tests.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SphericalGrid_test.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SphericalGrid_rayTrace_GPU_tests.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SpatialGrid_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SpatialGrid_Spherical_GPU_tester.cu
              )
target_link_libraries( monteray.geometry.test PRIVATE MonteRay::testing )
add_test(NAME monteray.geometry.test COMMAND monteray.geometry.test) 
create_lnk3dnt_symlink(monteray.geometry.test)

add_subdirectory(Cartesian)
add_subdirectory(Cylindrical)
