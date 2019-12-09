#include "MonteRayNextEventEstimator.t.hh"

#include "MonteRay_SpatialGrid.hh"
#include "MonteRay_GridBins.hh"

namespace MonteRay {

template class MonteRayNextEventEstimator<MonteRay_SpatialGrid>;
template class CopyMemoryBase<MonteRayNextEventEstimator<MonteRay_SpatialGrid>>;

}
