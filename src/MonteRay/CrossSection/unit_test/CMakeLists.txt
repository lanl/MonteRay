add_executable( monteray.crosssection.test )

setCudaFilesToCppFiles()

target_sources( monteray.crosssection.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/CrossSection_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/CrossSectionHash_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/CrossSectionList_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/Log2Hash_tester.cu
               ${CMAKE_CURRENT_LIST_DIR}/LogHash_tester.cu
              )
target_link_libraries( monteray.crosssection.test PRIVATE MonteRay::testing )
add_test(NAME monteray.crosssection.test COMMAND monteray.crosssection.test) 
create_MonteRayTestFiles_symlink(monteray.crosssection.test)

