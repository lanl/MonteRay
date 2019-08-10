add_executable( monteray.thirdparty.test )
setCudaFilesToCppFiles()
target_sources( monteray.thirdparty.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/ManagedAllocator_tester.cu
              )
target_link_libraries( monteray.thirdparty.test PRIVATE MonteRay::testing )
add_test(NAME monteray.thirdparty.test COMMAND monteray.thirdparty.test) 
