/*
 * MonteRayCartesianGrid.cc
 *
 *  Created on: Feb 2, 2018
 *      Author: jsweezy
 */

#include "MonteRay_CartesianGrid.hh"

namespace MonteRay {


#ifdef __CUDACC__
    CUDA_CALLABLE_KERNEL
    void createDeviceInstance(MonteRay_CartesianGrid** pPtrInstance, MonteRay_GridBins* pGridX, MonteRay_GridBins* pGridY, MonteRay_GridBins* pGridZ ) {
    		*pPtrInstance = new MonteRay_CartesianGrid( 3, pGridX, pGridY, pGridZ );
    		//if( debug ) printf( "Debug: createDeviceInstance -- pPtrInstance = %d\n", pPtrInstance );
    }

    CUDA_CALLABLE_KERNEL
    void deleteDeviceInstance(MonteRay_CartesianGrid* pInstance) {
    	delete pInstance;
    }
#endif

CUDA_CALLABLE_MEMBER
MonteRay_CartesianGrid::MonteRay_CartesianGrid(unsigned dim, pArrayOfpGridInfo_t pBins) :
    MonteRay_GridSystemInterface(dim)
{
	MONTERAY_VERIFY( dim == DimMax, "MonteRay_CartesianGrid::ctor -- only 3-D is allowed" ); // No greater than 3-D.

	DIM = 3;
	for(auto i = 0; i< dim; ++i) {
		pGridBins[i] = pBins[i];
	}
}

CUDA_CALLABLE_MEMBER
MonteRay_CartesianGrid::MonteRay_CartesianGrid(unsigned dim, GridBins_t* pGridX, GridBins_t* pGridY, GridBins_t* pGridZ ) :
    MonteRay_GridSystemInterface(dim)
{
	MONTERAY_VERIFY( dim == DimMax, "MonteRay_CartesianGrid::ctor -- only 3-D is allowed" ); // No greater than 3-D.

	DIM = 3;
	pGridBins[0] = pGridX;
	pGridBins[1] = pGridY;
	pGridBins[2] = pGridZ;
}



CUDA_CALLABLE_MEMBER
unsigned
MonteRay_CartesianGrid::getIndex( const GridBins_t::Position_t& particle_pos) const{
	if( debug ) printf("Debug: MonteRay_CartesianGrid::getIndex -- starting\n");

    int indices[3]= {0,0,0};
    for( auto d = 0; d < DIM; ++d ) {
    	if( debug ) printf("Debug: MonteRay_CartesianGrid::getIndex -- d = %d\n",d);
        indices[d] = getDimIndex(d, particle_pos[d] );

        // outside the grid
        if( isIndexOutside(d, indices[d] ) ) {
        	return UINT_MAX;
        }
    }

    if( debug ) printf("Debug: MonteRay_CartesianGrid::getIndex -- calling calcIndex\n");
    return calcIndex( indices );
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t
MonteRay_CartesianGrid::getVolume(unsigned index ) const {

    gpuFloatType_t volume=1.0;
//    if( regular ) {
//        for( unsigned d=0; d < DIM; ++d ) {
//            volume *= pGridBins[d]->delta;
//        }
//    } else {
    	uint3 indices = calcIJK( index );
    	volume *= pGridBins[0]->vertices[ indices.x + 1 ] - pGridBins[0]->vertices[ indices.x ];
    	volume *= pGridBins[1]->vertices[ indices.y + 1 ] - pGridBins[1]->vertices[ indices.y ];
    	volume *= pGridBins[2]->vertices[ indices.z + 1 ] - pGridBins[2]->vertices[ indices.z ];
//    }
    return volume;
}

CUDA_CALLABLE_MEMBER
unsigned
MonteRay_CartesianGrid::getNumBins( unsigned d) const {
	if( debug ) printf("Debug: MonteRay_CartesianGrid::getNumBins -- d= %d\n", d);
	if( debug ) printf("Debug: MonteRay_CartesianGrid::getNumBins --calling pGridBins[d]->getNumBins()\n");
	return pGridBins[d]->getNumBins();
}

CUDA_CALLABLE_MEMBER
unsigned
MonteRay_CartesianGrid::calcIndex( const int indices[] ) const{
    unsigned index = indices[0];
    if( DIM > 1 ) {
        index += indices[1]* getNumBins(0);
    }
    if( DIM > 2 ) {
        index += indices[2] * getNumBins(0)*getNumBins(1);
    }
    return index;
}

CUDA_CALLABLE_MEMBER
uint3
MonteRay_CartesianGrid::calcIJK( unsigned index ) const {
    uint3 indices;

    uint3 offsets;
    offsets.x = 1;

    offsets.y = getNumBins(0);
    offsets.z = getNumBins(0)*getNumBins(1);

    MONTERAY_ASSERT(offsets.z > 0 );
    MONTERAY_ASSERT(offsets.y > 0 );
    MONTERAY_ASSERT(offsets.x > 0 );

    indices.z = index / offsets.z;
    index -= indices.z * offsets.z;

    indices.y = index / offsets.y;
    index -= indices.y * offsets.y;

    indices.x = index / offsets.x;

    return indices;
}

CUDA_CALLABLE_MEMBER
bool
MonteRay_CartesianGrid::isOutside( const int i[] ) const {
    for( unsigned d=0; d<DIM; ++d){
       if( isIndexOutside(d, i[d]) ) return true;
    }
    return false;
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CartesianGrid::rayTrace( rayTraceList_t& rayTraceList, const GridBins_t::Position_t& particle_pos, const GridBins_t::Position_t& particle_direction, gpuFloatType_t distance,  bool outsideDistances) const{
	if( debug ) printf( "Debug: MonteRay_CartesianGrid::rayTrace -- \n");
	rayTraceList.reset();
    int indices[3] = {0, 0, 0}; // current position indices in the grid, must be int because can be outside

    multiDimRayTraceMap_t distances;
    for( unsigned d=0; d<DIM; ++d){
    	distances[d].reset();

        indices[d] = getDimIndex(d, particle_pos[d] );

        if( debug ) printf( "Debug: MonteRay_CartesianGrid::rayTrace -- dimension=%d, index=%d\n", d, indices[d]);

        planarCrossingDistance( distances[d],*(pGridBins[d]),particle_pos[d],particle_direction[d],distance,indices[d]);

        if( debug ) printf( "Debug: MonteRay_CartesianGrid::rayTrace -- dimension=%d, number of planar crossings = %d\n", d, distances[d].size() );

        // if outside and ray doesn't move inside then ray never enters the grid
        if( isIndexOutside(d,indices[d]) && distances[d].size() == 0  ) {
            return;
        }
    }

    orderCrossings( rayTraceList, distances, indices, distance, outsideDistances );

    if( debug ) printf( "Debug: MonteRay_CartesianGrid::rayTrace -- number of total crossings = %d\n", rayTraceList.size() );
    return;
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CartesianGrid::crossingDistance( singleDimRayTraceMap_t& rayTraceMap, unsigned d, gpuFloatType_t pos, gpuFloatType_t dir, gpuFloatType_t distance ) const {
    crossingDistance(rayTraceMap, *(pGridBins[d]), pos, dir, distance, false);
    return;
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CartesianGrid::crossingDistance( singleDimRayTraceMap_t& rayTraceMap, const GridBins_t& Bins, gpuFloatType_t pos, gpuFloatType_t dir, gpuFloatType_t distance, bool equal_spacing) const {
    int index = Bins.getLinearIndex(pos);
    planarCrossingDistance( rayTraceMap, Bins, pos, dir, distance, index);
    return;
}


} /* namespace MonteRay */
