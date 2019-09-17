target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/IPComm/MonteRayParallelAssistant.hh"
  "${prefix}/MonteRay/IPComm/MonteRay_SharedRayList.hh"
PRIVATE
  "${prefix}/MonteRay/IPComm/MonteRayParallelAssistant.cu"
)

if (MonteRay.tests)
  add_subdirectory(punit_test)
  add_subdirectory(unit_test)
endif()
