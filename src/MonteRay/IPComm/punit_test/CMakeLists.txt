add_executable( monteray.ipcomm.parallel.test )
target_sources( monteray.ipcomm.parallel.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/MonteRayParallelAssistant_ptester.cc
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SharedRayList_ptester.cc
              )
target_link_libraries( monteray.ipcomm.parallel.test PRIVATE MonteRay::testing )
add_test(NAME monteray.ipcomm.parallel.test.1 COMMAND mpirun --map-by node -np 1 monteray.ipcomm.parallel.test) 
add_test(NAME monteray.ipcomm.parallel.test.2 COMMAND mpirun --map-by node -np 2 monteray.ipcomm.parallel.test) 
add_test(NAME monteray.ipcomm.parallel.test.4 COMMAND mpirun --map-by node -np 4 monteray.ipcomm.parallel.test) 
