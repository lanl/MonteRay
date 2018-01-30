#ifndef MATERIAL_UNITTEST_HELPER_HH_
#define MATERIAL_UNITTEST_HELPER_HH_

#include "MonteRayDefinitions.hh"
#include "MonteRayConstants.hh"

using namespace MonteRay;

class MaterialTestHelper
{
public:

	MaterialTestHelper();

	~MaterialTestHelper();

	void setupTimers();

	void stopTimers();

private:
#ifdef __CUDACC__
	cudaEvent_t start, stop;
#endif

};
#endif /* MATERIAL_UNITTEST_HELPER_HH_ */


