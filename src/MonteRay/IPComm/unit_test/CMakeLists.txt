add_executable( monteray.ipcomm.test )
target_sources( monteray.ipcomm.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/MonteRayParallelAssistant_tester.cc
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SharedRayList_tester.cc
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_SharedNextEventRayList_tester.cc
              )
target_link_libraries( monteray.ipcomm.test PRIVATE MonteRay::testing )
add_test(NAME monteray.ipcomm.test COMMAND monteray.ipcomm.test) 
