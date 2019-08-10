add_executable( monteray.lnk3dnt.test )
target_sources( monteray.lnk3dnt.test  PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/main.cc
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_Check_lnk3dnt.cc
               ${CMAKE_CURRENT_LIST_DIR}/MonteRay_OpenReadParse.cc
              )
target_link_libraries( monteray.lnk3dnt.test PRIVATE MonteRay::testing )
add_test(NAME monteray.lnk3dnt.test COMMAND monteray.lnk3dnt.test) 

create_lnk3dnt_symlink(monteray.lnk3dnt.test)
