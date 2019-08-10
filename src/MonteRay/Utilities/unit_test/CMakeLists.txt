add_executable( monteray.unittest.test )
target_sources( monteray.unittest.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/UnitTest_test.cc
              )
target_link_libraries( monteray.unittest.test PRIVATE MonteRay::testing )
add_test(NAME monteray.unittest.test COMMAND monteray.unittest.test) 
