add_executable( monteray.SimpleVector.test )
setCudaFilesToCppFiles()
target_sources( monteray.SimpleVector.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/SimpleVector.test.cu
              )
target_link_libraries( monteray.SimpleVector.test PRIVATE MonteRay::testing )
add_test(NAME monteray.SimpleVector.test COMMAND monteray.SimpleVector.test) 

add_executable( monteray.SimpleView.test )
setCudaFilesToCppFiles()
target_sources( monteray.SimpleView.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/SimpleView.test.cu
              )
target_link_libraries( monteray.SimpleView.test PRIVATE MonteRay::testing )
add_test(NAME monteray.SimpleView.test COMMAND monteray.SimpleView.test) 
