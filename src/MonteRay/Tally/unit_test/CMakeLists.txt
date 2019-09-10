add_executable( monteray.monteraytally.test )
setCudaFilesToCppFiles()
target_sources( monteray.monteraytally.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/MonteRayNextEventEstimator_unittest.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRayTally_GPU_test_helper.hh
               ${CMAKE_CURRENT_LIST_DIR}/MonteRayTally_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRayTally_GPU_tester.cu
              )
target_link_libraries( monteray.monteraytally.test PRIVATE MonteRay::testing )
add_test(NAME monteray.monteraytally.test COMMAND monteray.monteraytally.test) 

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
