add_executable( monteray.raytrace.test )
setCudaFilesToCppFiles()
target_sources( monteray.raytrace.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/genericGPU_test_helper.hh
               ${CMAKE_CURRENT_LIST_DIR}/genericGPU_test_helper.cu
               ${CMAKE_CURRENT_LIST_DIR}/gpuDistanceCalculator_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/gpuDistanceCalculator_test_helper.hh
               ${CMAKE_CURRENT_LIST_DIR}/gpuDistanceCalculator_test_helper.cu
               ${CMAKE_CURRENT_LIST_DIR}/gpuTiming_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/gpuTiming_test_helper.cu
               ${CMAKE_CURRENT_LIST_DIR}/gpuTiming_test_helper.hh
               ${CMAKE_CURRENT_LIST_DIR}/RayListController_unit_tester.cu
              )
target_link_libraries( monteray.raytrace.test PRIVATE MonteRay::testing )
add_test(NAME monteray.raytrace.test COMMAND monteray.raytrace.test) 

