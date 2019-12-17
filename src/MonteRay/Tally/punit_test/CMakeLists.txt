add_executable( monteray.tally.parallel.test )
setCudaFilesToCppFiles()
target_sources( monteray.tally.parallel.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/NextEventEstimator_ptester.cu
               ${CMAKE_CURRENT_LIST_DIR}/Tally_ptester.cu
              )
target_link_libraries( monteray.tally.parallel.test PRIVATE MonteRay::testing )
add_test(NAME monteray.tally.parallel.test.1 COMMAND mpirun --map-by node -np 1 monteray.tally.parallel.test) 
add_test(NAME monteray.tally.parallel.test.4 COMMAND mpirun --map-by node -np 4 monteray.tally.parallel.test) 
