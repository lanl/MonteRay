add_executable( monteray.raytrace.fi.test )
setCudaFilesToCppFiles()
target_sources( monteray.raytrace.fi.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/RayListController_wNextEventEstimator_fi_tester.cu
              )
target_link_libraries( monteray.raytrace.fi.test PRIVATE MonteRay::testing )
add_test(NAME monteray.raytrace.fi.test COMMAND monteray.raytrace.fi.test) 
create_MonteRayTestFiles_symlink(monteray.raytrace.fi.test)
