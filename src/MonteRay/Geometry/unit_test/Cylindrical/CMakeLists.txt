add_executable( monteray.geometry.cylindrical.test )
setCudaFilesToCppFiles()
target_sources( monteray.geometry.cylindrical.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/../main.cc
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_CylindricalGrid_test.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_CylindricalGrid_crossingDistance_tests.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_CylindricalGrid_rayTrace_tests.cu
              )

target_link_libraries( monteray.geometry.cylindrical.test PRIVATE MonteRay::testing )
add_test(NAME monteray.geometry.cylindrical.test COMMAND monteray.geometry.cylindrical.test) 
create_lnk3dnt_symlink(monteray.geometry.cylindrical.test)
