/*
 * MonteRaySphericalGrid.cc
 *
 *  Created on: Feb 2, 2018
 *      Author: jsweezy
 */

#include "MonteRay_SphericalGrid.hh"
#include "MonteRayConstants.hh"

#include <float.h>

namespace MonteRay {

#ifdef __CUDACC__
    CUDA_CALLABLE_KERNEL
    void createDeviceInstance(MonteRay_SphericalGrid** pPtrInstance, ptrSphericalGrid_result_t* pResult, MonteRay_GridBins* pGridR ) {
    		*pPtrInstance = new MonteRay_SphericalGrid( 1, pGridR );
    		pResult->v = *pPtrInstance;
    		//if( debug ) printf( "Debug: createDeviceInstance -- pPtrInstance = %d\n", pPtrInstance );
    }

    CUDA_CALLABLE_KERNEL
    void deleteDeviceInstance(MonteRay_SphericalGrid** pPtrInstance) {
    	delete *pPtrInstance;
    }

    CUDAHOST_CALLABLE_MEMBER
    MonteRay_SphericalGrid*
    MonteRay_SphericalGrid::getDeviceInstancePtr() {
    	return devicePtr;
    }

#endif

CUDA_CALLABLE_MEMBER
MonteRay_SphericalGrid::MonteRay_SphericalGrid(unsigned dim, pArrayOfpGridInfo_t pBins) :
    MonteRay_GridSystemInterface(dim)
{
	MONTERAY_VERIFY( dim == DimMax, "MonteRay_SphericalGrid::ctor -- only 1-D is allowed" ); // No greater than 1-D.

	DIM = 1;
	pRVertices = pBins[0];
	validate();
}

CUDA_CALLABLE_MEMBER
MonteRay_SphericalGrid::MonteRay_SphericalGrid(unsigned dim, GridBins_t* pGridR ) :
    MonteRay_GridSystemInterface(dim)
{
	MONTERAY_VERIFY( dim == DimMax, "MonteRay_SphericalGrid::ctor -- only 1-D is allowed" ); // No greater than 1-D.

	DIM = 1;
	pRVertices = pGridR;
	validate();
}


CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::validate(void) {
    validateR();
    numRBins = pRVertices->getNumBins();
}

CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::validateR(void) {
    // Test for negative R
	for( int i=0; i<pRVertices->nVertices; ++i ){
        MONTERAY_VERIFY( pRVertices->vertices[i] >= 0.0, "MonteRay_SphericalGrid::validateR -- Can't have negative values for radius!!!" );
    }

    pRVertices->modifyForRadial();
}

CUDA_CALLABLE_MEMBER
MonteRay_SphericalGrid::Position_t
MonteRay_SphericalGrid::convertFromCartesian( const Position_t& pos) const {
     Position_t particleMeshPosition = {0.0, 0.0, 0.0};

     gpuRayFloat_t r = sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
     particleMeshPosition[R] = r;

     return particleMeshPosition;
}


CUDA_CALLABLE_MEMBER
unsigned
MonteRay_SphericalGrid::getIndex( const Position_t& particle_pos) const{
	if( debug ) printf("Debug: MonteRay_SphericalGrid::getIndex -- starting\n");

    int index = 0;
    Position_t pos = convertFromCartesian( particle_pos );

    printf("%i\n", pRVertices->isRadial() );
    index = pRVertices->getRadialIndexFromR( pos[R] );
    if( isIndexOutside(R, index ) ) { return UINT_MAX; }

    return index;
}

CUDA_CALLABLE_MEMBER
bool
MonteRay_SphericalGrid::isOutside( const int i[] ) const {
	if( isIndexOutside(R, i[R]) ) { return true; }
    return false;
}

CUDA_CALLABLE_MEMBER
unsigned
MonteRay_SphericalGrid::calcIndex( const int indices[] ) const{
    return indices[R];
}


CUDA_CALLABLE_MEMBER
gpuRayFloat_t
MonteRay_SphericalGrid::getVolume( unsigned index ) const {
	gpuRayFloat_t innerRadius = 0.0;
    if( index > 0 ){
        innerRadius = pRVertices->vertices[index-1];
    }
    gpuRayFloat_t outerRadius = pRVertices->vertices[index];

    gpuRayFloat_t volume = 4.0 * MonteRay::pi * ( std::pow(outerRadius,3) - std::pow(innerRadius,3) ) / 3.0 ;

    return volume;
}

CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::rayTrace( rayTraceList_t& rayTraceList, const GridBins_t::Position_t&, const GridBins_t::Position_t&, gpuRayFloat_t distance,  bool outsideDistances/*=false*/) const {

}

CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::crossingDistance(singleDimRayTraceMap_t& rayTraceMap, unsigned d, gpuRayFloat_t pos, gpuRayFloat_t dir, gpuRayFloat_t distance ) const {

}

CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::crossingDistance(singleDimRayTraceMap_t& rayTraceMap, const GridBins_t& Bins, gpuRayFloat_t pos, gpuRayFloat_t dir, gpuRayFloat_t distance, bool equal_spacing/*=false*/) const {

}

CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::radialCrossingDistances(singleDimRayTraceMap_t& rayTraceMap, const Position_t& pos, const Direction_t& dir, unsigned rIndex, gpuRayFloat_t distance ) const {

}

CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::radialCrossingDistances( singleDimRayTraceMap_t& rayTraceMap, const Position_t& pos, const Direction_t& dir, gpuRayFloat_t distance ) const {

}

CUDA_CALLABLE_MEMBER
void
MonteRay_SphericalGrid::radialCrossingDistancesSingleDirection( singleDimRayTraceMap_t& rayTraceMap, const Position_t& pos, const Direction_t& dir, gpuRayFloat_t distance, bool outward ) const {

}

//CUDA_CALLABLE_MEMBER
//void
//MonteRay_SphericalGrid::rayTrace( rayTraceList_t& rayTraceList, const GridBins_t::Position_t& particle_pos, const GridBins_t::Position_t& particle_direction, gpuFloatType_t distance,  bool outsideDistances) const{
//	if( debug ) printf( "Debug: MonteRay_SphericalGrid::rayTrace -- \n");
//	rayTraceList.reset();
//    int indices[3] = {0, 0, 0}; // current position indices in the grid, must be int because can be outside
//
//    multiDimRayTraceMap_t distances;
//    for( unsigned d=0; d<DIM; ++d){
//    	distances[d].reset();
//
//        indices[d] = getDimIndex(d, particle_pos[d] );
//
//        if( debug ) printf( "Debug: MonteRay_SphericalGrid::rayTrace -- dimension=%d, index=%d\n", d, indices[d]);
//
//        planarCrossingDistance( distances[d],*(pGridBins[d]),particle_pos[d],particle_direction[d],distance,indices[d]);
//
//        if( debug ) printf( "Debug: MonteRay_SphericalGrid::rayTrace -- dimension=%d, number of planar crossings = %d\n", d, distances[d].size() );
//
//        // if outside and ray doesn't move inside then ray never enters the grid
//        if( isIndexOutside(d,indices[d]) && distances[d].size() == 0  ) {
//            return;
//        }
//    }
//
//    orderCrossings( rayTraceList, distances, indices, distance, outsideDistances );
//
//    if( debug ) printf( "Debug: MonteRay_SphericalGrid::rayTrace -- number of total crossings = %d\n", rayTraceList.size() );
//    return;
//}
//
//CUDA_CALLABLE_MEMBER
//void
//MonteRay_SphericalGrid::crossingDistance( singleDimRayTraceMap_t& rayTraceMap, unsigned d, gpuFloatType_t pos, gpuFloatType_t dir, gpuFloatType_t distance ) const {
//    crossingDistance(rayTraceMap, *(pGridBins[d]), pos, dir, distance, false);
//    return;
//}
//
//CUDA_CALLABLE_MEMBER
//void
//MonteRay_SphericalGrid::crossingDistance( singleDimRayTraceMap_t& rayTraceMap, const GridBins_t& Bins, gpuFloatType_t pos, gpuFloatType_t dir, gpuFloatType_t distance, bool equal_spacing) const {
//    int index = Bins.getLinearIndex(pos);
//    planarCrossingDistance( rayTraceMap, Bins, pos, dir, distance, index);
//    return;
//}


} /* namespace MonteRay */
