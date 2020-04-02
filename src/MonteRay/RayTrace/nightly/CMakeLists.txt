add_executable( monteray.raytrace_assembly.nightly.test )
setCudaFilesToCppFiles()
target_sources( monteray.raytrace_assembly.nightly.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/PWR_Assembly_CollisionsFromFile_nightly_tester.cu
              )
target_link_libraries( monteray.raytrace_assembly.nightly.test PRIVATE MonteRay::testing )
add_test(NAME monteray.raytrace_assembly.nightly.test COMMAND monteray.raytrace_assembly.nightly.test) 
create_MonteRayTestFiles_symlink(monteray.raytrace_assembly.nightly.test)
create_lnk3dnt_symlink(monteray.raytrace_assembly.nightly.test)

add_executable( monteray.raytrace_criticality.nightly.test )
target_sources( monteray.raytrace_criticality.nightly.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/Criticality_Accident_CollisionsFromFile_nightly_tester.cu
              )
target_link_libraries( monteray.raytrace_criticality.nightly.test PRIVATE MonteRay::testing )
add_test(NAME monteray.raytrace_criticality.nightly.test COMMAND monteray.raytrace_criticality.nightly.test) 
create_MonteRayTestFiles_symlink(monteray.raytrace_criticality.nightly.test)
create_lnk3dnt_symlink(monteray.raytrace_criticality.nightly.test)

add_executable( monteray.raytrace_zeus.nightly.test )
target_sources( monteray.raytrace_zeus.nightly.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/Zeus2_Cylindrical_CollisionsFromFile_nightly_tester.cu
              )
target_link_libraries( monteray.raytrace_zeus.nightly.test PRIVATE MonteRay::testing )
add_test(NAME monteray.raytrace_zeus.nightly.test COMMAND monteray.raytrace_zeus.nightly.test) 
create_MonteRayTestFiles_symlink(monteray.raytrace_zeus.nightly.test)
create_lnk3dnt_symlink(monteray.raytrace_zeus.nightly.test)

