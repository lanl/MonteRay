add_executable( monteray.CudaWrappers.test )
setCudaFilesToCppFiles()
target_sources( monteray.CudaWrappers.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/cudaWrappers.test.cu
              )
target_link_libraries( monteray.CudaWrappers.test PRIVATE MonteRay::testing )
add_test(NAME monteray.CudaWrappers.test COMMAND monteray.CudaWrappers.test) 
