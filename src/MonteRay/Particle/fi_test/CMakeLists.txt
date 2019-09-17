add_executable( monteray.particle.fi.test )
setCudaFilesToCppFiles()
target_sources( monteray.particle.fi.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/fi_genericGPU_test_helper.hh
               ${CMAKE_CURRENT_LIST_DIR}/fi_genericGPU_test_helper.cu
               ${CMAKE_CURRENT_LIST_DIR}/RayListInterface_fi_tester.cu
              )
target_link_libraries( monteray.particle.fi.test PRIVATE MonteRay::testing )
add_test(NAME monteray.particle.fi.test COMMAND monteray.particle.fi.test) 
create_MonteRayTestFiles_symlink(monteray.particle.fi.test)
create_lnk3dnt_symlink(monteray.particle.fi.test)
