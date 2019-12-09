#include "MonteRay_SpatialGrid.hh"
#include "MonteRay_CartesianGrid.t.hh"
#include "MonteRay_CylindricalGrid.t.hh"
/* #include "MonteRay_SphericalGrid.t.hh" */
#include "MonteRay_binaryIO.hh"
#include "MonteRayCopyMemory.t.hh"
#include "MonteRay_GridSystemInterface.hh"
#include "RayWorkInfo.hh"
#include "MonteRayParallelAssistant.hh"

#include <stdexcept>
#include <sstream>
#include <fstream>


namespace MonteRay {


// TPB TODO: revisit the need for this
void MonteRay_SpatialGrid::checkDim( unsigned dim ) const {
  unsigned maxDim = dimension() > 0 ? dimension() : MaxDim;
  if( dim > maxDim ) {
    std::stringstream msg;
    msg << " Dimension greater than MaxDim = " << maxDim << "!!! \n"
            << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_SpatialGrid::checkDim \n\n";
    throw std::runtime_error( msg.str() );
  }
}

/* void MonteRay_SpatialGrid::write(std::ostream& outf) const { */
/*   unsigned version = 0; */
/*   binaryIO::write(outf, version ); */

/*   binaryIO::write(outf, coordinateSystem); */
/* } */

/* MonteRay_SpatialGrid MonteRay_SpatialGrid::read(std::istream& infile) { */
/*   unsigned version = 0; */
/*   binaryIO::read(infile, version ); */

/*   TransportMeshType coordinateSystem; */
/*   binaryIO::read(infile, coordinateSystem); */
  
/* } */

} /* namespace MonteRay */
