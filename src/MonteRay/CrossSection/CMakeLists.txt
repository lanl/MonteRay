target_sources(MonteRay 
PUBLIC 
  "${prefix}/MonteRay/CrossSection/CrossSectionUtilities.hh"
  "${prefix}/MonteRay/CrossSection/FasterHash.hh"
  "${prefix}/MonteRay/CrossSection/CrossSectionHash.hh"
  "${prefix}/MonteRay/CrossSection/CrossSection.hh"
  "${prefix}/MonteRay/CrossSection/CrossSectionList.hh"
PRIVATE
  "${prefix}/MonteRay/CrossSection/CrossSectionUtilities.cu"
  "${prefix}/MonteRay/CrossSection/CrossSectionHash.cu"
  "${prefix}/MonteRay/CrossSection/CrossSection.cu"
  "${prefix}/MonteRay/CrossSection/CrossSectionList.cu"
)

if (MonteRay.tests)
  add_subdirectory(unit_test)
endif()











