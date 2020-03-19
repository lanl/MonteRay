add_executable( monteray.material.test )
setCudaFilesToCppFiles()
target_sources( monteray.material.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/MaterialList_tester.cu
              )
target_link_libraries( monteray.material.test PRIVATE MonteRay::testing )
add_test(NAME monteray.material.test COMMAND monteray.material.test) 
create_MonteRayTestFiles_symlink(monteray.material.test)


add_executable( monteray.material_list.test )
setCudaFilesToCppFiles()
target_sources( monteray.material_list.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/MaterialList_tester.cu
              )
target_link_libraries( monteray.material_list.test PRIVATE MonteRay::testing )
add_test(NAME monteray.material_list.test COMMAND monteray.material_list.test) 
create_MonteRayTestFiles_symlink(monteray.material_list.test)
