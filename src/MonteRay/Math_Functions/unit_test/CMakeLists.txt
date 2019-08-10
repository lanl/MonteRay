add_executable( monteray.mathfunctions.test )
target_sources( monteray.mathfunctions.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/BinarySearch_test.cc
               ${CMAKE_CURRENT_LIST_DIR}/LinearySearch_test.cc
               ${CMAKE_CURRENT_LIST_DIR}/Vector3D_test.cc
              )
target_link_libraries( monteray.mathfunctions.test PRIVATE MonteRay::testing )
add_test(NAME monteray.mathfunctions.test COMMAND monteray.mathfunctions.test) 
