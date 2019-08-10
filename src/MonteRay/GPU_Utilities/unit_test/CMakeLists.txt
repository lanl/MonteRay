add_executable( monteray.gpuutilities.test )
setCudaFilesToCppFiles()
target_sources( monteray.gpuutilities.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/CopyMemory_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/GPUAtomicAdd_unittest.cu
               ${CMAKE_CURRENT_LIST_DIR}/GPUDeviceProperties_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/GPUUtilityFunctions_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/ManagedMemoryAllocator_Vector_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/ManagedMemory_test_helper.hh
               ${CMAKE_CURRENT_LIST_DIR}/ManagedMemory_test_helper.cu
               ${CMAKE_CURRENT_LIST_DIR}/ManagedMemory_tester.cu
              )
target_link_libraries( monteray.gpuutilities.test PRIVATE MonteRay::testing )
add_test(NAME monteray.gpuutilities.test COMMAND monteray.gpuutilities.test) 
