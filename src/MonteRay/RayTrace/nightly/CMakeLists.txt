add_executable( monteray.raytrace.nightly.test )
target_sources( monteray.raytrace.nightly.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/Criticality_Accident_CollisionsFromFile_nightly_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/PWR_Assembly_CollisionsFromFile_nightly_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/RayTraceNightly_genericGPU_test_helper.hh
               ${CMAKE_CURRENT_LIST_DIR}/RayTraceNightly_genericGPU_test_helper.cu
               ${CMAKE_CURRENT_LIST_DIR}/Zeus2_Cylindrical_CollisionsFromFile_nightly_tester.cu
              )
target_link_libraries( monteray.raytrace.nightly.test PRIVATE MonteRay::testing )
add_test(NAME monteray.raytrace.nightly.test COMMAND monteray.raytrace.nightly.test) 
create_MonteRayTestFiles_symlink(monteray.raytrace.nightly.test)
create_lnk3dnt_symlink(monteray.raytrace.nightly.test)
