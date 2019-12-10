add_executable( monteray.geometry.test )
setCudaFilesToCppFiles()
target_sources( monteray.geometry.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/MaterialProperties_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_GridBins_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_GridSystemInterface_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SpatialGrid_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_GridVariant_tester.cu
              )
target_link_libraries( monteray.geometry.test PRIVATE MonteRay::testing )
add_test(NAME monteray.geometry.test COMMAND monteray.geometry.test) 
create_lnk3dnt_symlink(monteray.geometry.test)

add_subdirectory(Cartesian)
# add_subdirectory(Cylindrical)
