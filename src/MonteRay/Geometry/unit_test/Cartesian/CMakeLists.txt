add_executable( monteray.geometry.cartesian.test )
setCudaFilesToCppFiles()
target_sources( monteray.geometry.cartesian.test  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../main.cc
    ${CMAKE_CURRENT_LIST_DIR}/MonteRay_CartesianGrid_rayTrace_tests.cu
    ${CMAKE_CURRENT_LIST_DIR}/MonteRay_CartesianGrid_crossingDistance_tests.cu
    ${CMAKE_CURRENT_LIST_DIR}/MonteRay_CartesianGrid_rayTrace_GPU_tests.cu
    ${CMAKE_CURRENT_LIST_DIR}/MonteRay_CartesianGrid_test.cu
    ${CMAKE_CURRENT_LIST_DIR}/MonteRay_CartesianGrid_crossingDistance_GPU_tests.cu
    ${CMAKE_CURRENT_LIST_DIR}/MonteRay_CartesianGrid_rayTraceWithMovingMaterials_tests.cu
    ${CMAKE_CURRENT_LIST_DIR}/../RayTraceHelper.hh
    ${CMAKE_CURRENT_LIST_DIR}/../CrossingDistanceHelper.hh
              )

target_link_libraries( monteray.geometry.cartesian.test PRIVATE MonteRay::testing )
add_test(NAME monteray.geometry.cartesian.test COMMAND monteray.geometry.cartesian.test) 
create_lnk3dnt_symlink(monteray.geometry.cartesian.test)
