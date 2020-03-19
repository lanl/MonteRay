add_executable( monteray.raytrace.parallel.fi.test )
setCudaFilesToCppFiles()
target_sources( monteray.raytrace.parallel.fi.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/RayListController_wNextEventEstimator_pfi_tester.cu
              )
target_link_libraries( monteray.raytrace.parallel.fi.test PRIVATE MonteRay::testing )
add_test(NAME monteray.raytrace.parallel.fi.test.1 COMMAND mpirun --map-by node -np 1 monteray.raytrace.parallel.fi.test) 
add_test(NAME monteray.raytrace.parallel.fi.test.4 COMMAND mpirun --map-by node -np 4 monteray.raytrace.parallel.fi.test) 
