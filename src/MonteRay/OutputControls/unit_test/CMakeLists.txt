add_executable( monteray.multistream.test )
target_sources( monteray.multistream.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/MultiStream_test.cc
              )
target_link_libraries( monteray.multistream.test PRIVATE MonteRay::testing )
add_test(NAME monteray.multistream.test COMMAND monteray.multistream.test) 
