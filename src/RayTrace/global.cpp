#include "global.h"

#include <limits>

namespace MonteRay{

const global::float_t global::epsilon = std::numeric_limits<double>::epsilon();
const global::float_t global::inf = std::numeric_limits<double>::infinity();

}
