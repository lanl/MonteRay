add_executable( monteray.tally.parallel.test )
setCudaFilesToCppFiles()
target_sources( monteray.tally.parallel.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/MonteRayNextEventEstimator_ptester.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRayTally_ptester.cu
              )
target_link_libraries( monteray.tally.parallel.test PRIVATE MonteRay::testing )
add_test(NAME monteray.tally.parallel.test COMMAND monteray.tally.parallel.test) 
