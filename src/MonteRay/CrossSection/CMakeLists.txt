target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/CrossSection/MonteRayCrossSection.hh"
  "${prefix}/MonteRay/CrossSection/CrossSectionUtilities.hh"
  "${prefix}/MonteRay/CrossSection/HashLookup.hh"
  "${prefix}/MonteRay/CrossSection/FasterHash.hh"
  "${prefix}/MonteRay/CrossSection/CrossSectionHash.hh"
  "${prefix}/MonteRay/CrossSection/CrossSection.hh"
  "${prefix}/MonteRay/CrossSection/CrossSectionList.hh"
PRIVATE
  "${prefix}/MonteRay/CrossSection/HashLookup.cu"
  "${prefix}/MonteRay/CrossSection/CrossSectionUtilities.cu"
  "${prefix}/MonteRay/CrossSection/CrossSectionHash.cu"
  "${prefix}/MonteRay/CrossSection/CrossSection.cu"
  "${prefix}/MonteRay/CrossSection/MonteRayCrossSection.cu"
  "${prefix}/MonteRay/CrossSection/CrossSectionList.cu"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
endif()











