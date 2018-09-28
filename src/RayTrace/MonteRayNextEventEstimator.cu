#include "MonteRayNextEventEstimator.t.hh"

#include "MonteRay_SpatialGrid.hh"
#include "MonteRay_GridBins.hh"

namespace MonteRay {

template class MonteRayNextEventEstimator<GridBins>;
template class MonteRayNextEventEstimator<MonteRay_SpatialGrid>;
template class CopyMemoryBase<MonteRayNextEventEstimator<GridBins> >;
template class CopyMemoryBase<MonteRayNextEventEstimator<MonteRay_SpatialGrid>>;

}
