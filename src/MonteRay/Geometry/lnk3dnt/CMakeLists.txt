
target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/Geometry/lnk3dnt/MonteRay_BinaryReadFcns.hh"
  "${prefix}/MonteRay/Geometry/lnk3dnt/MonteRay_ByteReverser.hh"
  "${prefix}/MonteRay/Geometry/lnk3dnt/MonteRay_ReadLnk3dnt.hh"
  "${prefix}/MonteRay/Geometry/lnk3dnt/MonteRay_BinaryWriteFcns.hh"
PRIVATE
  "${prefix}/MonteRay/Geometry/lnk3dnt/MonteRay_BinaryFileFcn.cc"
  "${prefix}/MonteRay/Geometry/lnk3dnt/MonteRay_ReadLnk3dnt.cc"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
endif()


