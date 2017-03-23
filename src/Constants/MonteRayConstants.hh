#ifndef MONTERAYCONSTANTS_HH_
#define MONTERAYCONSTANTS_HH_

#include <limits>
#include "MonteRayDefinitions.hh"

namespace MonteRay{

// Constants
const float_t epsilon = std::numeric_limits<double>::epsilon();
const float_t inf = std::numeric_limits<double>::infinity();

const gpuFloatType_t gpu_neutron_molar_mass = 1.00866491597f;
const gpuFloatType_t gpu_AvogadroBarn = .602214179f;

}



#endif /* MONTERAYCONSTANTS_HH_ */
