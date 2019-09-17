add_executable( monteray.ipcomm.parallel.test )
target_sources( monteray.ipcomm.parallel.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/MonteRayParallelAssistant_ptester.cc
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SharedRayList_ptester.cc
              )
target_link_libraries( monteray.ipcomm.parallel.test PRIVATE MonteRay::testing )
add_test(NAME monteray.ipcomm.parallel.test COMMAND monteray.ipcomm.parallel.test) 
