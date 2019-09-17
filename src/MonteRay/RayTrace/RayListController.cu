
#include "RayListController.t.hh"
#include "MonteRay_SpatialGrid.hh"
#include "GridBins.hh"

namespace MonteRay {

template class RayListController<GridBins,1>;
template class RayListController<GridBins,3>;
template class RayListController<MonteRay_SpatialGrid,1>;
template class RayListController<MonteRay_SpatialGrid,3>;

} // end namespace MonteRay
