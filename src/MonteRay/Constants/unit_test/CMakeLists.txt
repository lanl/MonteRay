add_executable( monteray.constants.test )
target_sources( monteray.constants.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/Constants_test.cc
              )
target_link_libraries( monteray.constants.test PRIVATE MonteRay::testing )
add_test(NAME monteray.constants.test COMMAND monteray.constants.test) 
