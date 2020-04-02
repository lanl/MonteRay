#ifndef MONTERAYCONSTANTS_HH_
#define MONTERAYCONSTANTS_HH_

#include <limits>
#include "MonteRayTypes.hh"
#include "ThirdParty/Math.hh"

#ifndef __CUDACC__
#include <cmath>
#endif

namespace MonteRay{

// Constants
constexpr float_t epsilon = std::numeric_limits<double>::epsilon();
constexpr float_t inf = std::numeric_limits<float_t>::infinity();

constexpr gpuFloatType_t neutron_molar_mass = 1.00866491597f;
constexpr gpuFloatType_t gpu_neutron_molar_mass = neutron_molar_mass;
constexpr gpuFloatType_t AvogadroBarn = .602214179f;
constexpr gpuFloatType_t gpu_AvogadroBarn = AvogadroBarn;

/// PI from CRC Standard Mathematical Tables, 28th Edition, Editor William H. Beyer, CRC Press, 1987
constexpr gpuFloatType_t pi = 3.14159265358979323846264338;

constexpr ParticleType_t neutron = 0;
constexpr ParticleType_t photon  = 1;

/// speed of light in a vaccume from ( http://physics.nist.gov/cgi-bin/cuu/Value?c ) Units [cm/shake]. Exact, according to NIST
constexpr gpuFloatType_t speed_of_light = 299.7924580; // [cm/shake]

/// neutron rest mass energy in MeV from ( http://physics.nist.gov/cgi-bin/cuu/Value?mnc2mev ) in Units [MeV]. Standard Uncertainty = 0.000 023 MeV
constexpr gpuFloatType_t neutron_rest_mass_MeV = 939.565346; // [MeV]

/// Returns the velocity of a neutron in cm/shake from neutron's energy in [MeV]
CUDA_CALLABLE_MEMBER
inline gpuFloatType_t neutron_speed_from_energy_const(){ return speed_of_light * Math::sqrt( static_cast<gpuFloatType_t>(2.0) / neutron_rest_mass_MeV );} // [MeV]
CUDA_CALLABLE_MEMBER
inline gpuFloatType_t inv_neutron_speed_from_energy_const(){ return static_cast<gpuFloatType_t>(1.0)/neutron_speed_from_energy_const(); }

}



#endif /* MONTERAYCONSTANTS_HH_ */
