setCudaFilesToCppFiles()

add_executable( monteray.tally.test )
target_sources( monteray.tally.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/Tally_tester.cu
              )
target_link_libraries( monteray.tally.test PRIVATE MonteRay::testing )
add_test(NAME monteray.tally.test COMMAND monteray.tally.test) 

add_executable( monteray.NextEventEstimator.test )
target_sources( monteray.NextEventEstimator.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/NextEventEstimator_tester.cu
              )
target_link_libraries( monteray.NextEventEstimator.test PRIVATE MonteRay::testing )
add_test(NAME monteray.NextEventEstimator.test COMMAND monteray.NextEventEstimator.test) 

add_executable( monteray.filter.test )
target_sources( monteray.filter.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/Filter_tester.cu
              )
target_link_libraries( monteray.filter.test PRIVATE MonteRay::testing )
add_test(NAME monteray.filter.test COMMAND monteray.filter.test) 

add_executable( monteray.rayTraceWithMovingMaterials.test )
target_sources( monteray.rayTraceWithMovingMaterials.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/RayTraceOnGridWithMovingMaterials_tester.cu
              )
target_link_libraries( monteray.rayTraceWithMovingMaterials.test PRIVATE MonteRay::testing )
add_test(NAME monteray.rayTraceWithMovingMaterials.test COMMAND monteray.rayTraceWithMovingMaterials.test) 
