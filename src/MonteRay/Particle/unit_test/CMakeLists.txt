add_executable( monteray.particle.test )
setCudaFilesToCppFiles()
target_sources( monteray.particle.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/RayListInterface_tester.cc
               ${CMAKE_CURRENT_LIST_DIR}/RayListInterface_test_helper.hh
               ${CMAKE_CURRENT_LIST_DIR}/RayListInterface_test_helper.cc
               ${CMAKE_CURRENT_LIST_DIR}/RayList_tester.cc
               ${CMAKE_CURRENT_LIST_DIR}/Ray_tester.cc
               ${CMAKE_CURRENT_LIST_DIR}/RayWorkInfo_tester.cc
              )
target_link_libraries( monteray.particle.test PRIVATE MonteRay::testing )
add_test(NAME monteray.particle.test COMMAND monteray.particle.test) 
create_MonteRayTestFiles_symlink(monteray.particle.test)
