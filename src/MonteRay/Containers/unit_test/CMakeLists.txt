add_executable( monteray.containers.test )
setCudaFilesToCppFiles()
target_sources( monteray.containers.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/SimpleVector.test.cu
              )
target_link_libraries( monteray.containers.test PRIVATE MonteRay::testing )
add_test(NAME monteray.containers.test COMMAND monteray.containers.test) 
