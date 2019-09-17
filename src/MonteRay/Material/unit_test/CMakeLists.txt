add_executable( monteray.material.test )
setCudaFilesToCppFiles()
target_sources( monteray.material.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/Material_test_helper.hh
               ${CMAKE_CURRENT_LIST_DIR}/Material_test_helper.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRayMaterialList_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/MonteRayMaterial_tester.cu
              )
target_link_libraries( monteray.material.test PRIVATE MonteRay::testing )
add_test(NAME monteray.material.test COMMAND monteray.material.test) 
create_MonteRayTestFiles_symlink(monteray.material.test)
