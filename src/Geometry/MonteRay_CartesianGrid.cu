/*
 * MonteRayCartesianGrid.cc
 *
 *  Created on: Feb 2, 2018
 *      Author: jsweezy
 */

#include "MonteRay_CartesianGrid.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRay_SingleValueCopyMemory.t.hh"
#include "MonteRayCopyMemory.t.hh"
#include "RayWorkInfo.hh"
#include "MonteRayParallelAssistant.hh"

#include <float.h>

namespace MonteRay {

using ptrCartesianGrid_result_t = MonteRay_SingleValueCopyMemory<MonteRay_CartesianGrid*>;

CUDA_CALLABLE_KERNEL  createDeviceInstance(MonteRay_CartesianGrid** pPtrInstance, ptrCartesianGrid_result_t* pResult, MonteRay_GridBins* pGridX, MonteRay_GridBins* pGridY, MonteRay_GridBins* pGridZ ) {
    *pPtrInstance = new MonteRay_CartesianGrid( 3, pGridX, pGridY, pGridZ );
    pResult->v = *pPtrInstance;
    //if( debug ) printf( "Debug: createDeviceInstance -- pPtrInstance = %d\n", pPtrInstance );
}

CUDA_CALLABLE_KERNEL  deleteDeviceInstance(MonteRay_CartesianGrid** pPtrInstance) {
    delete *pPtrInstance;
}

CUDAHOST_CALLABLE_MEMBER
MonteRay_CartesianGrid*
MonteRay_CartesianGrid::getDeviceInstancePtr() {
    return devicePtr;
}


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
MonteRay_CartesianGrid::~MonteRay_CartesianGrid(void){
#ifdef __CUDACC__
#ifndef __CUDA_ARCH__
    if( ptrDevicePtr ) {
        deleteDeviceInstance<<<1,1>>>( ptrDevicePtr );
        cudaDeviceSynchronize();
    }
    MonteRayDeviceFree( ptrDevicePtr );
#endif
#endif
}

CUDAHOST_CALLABLE_MEMBER
void
MonteRay_CartesianGrid::copyToGPU(void) {
    if( debug ) std::cout << "Debug: MonteRay_CartesianGrid::copyToGPU \n";
#ifdef __CUDACC__
    if( ! MonteRay::isWorkGroupMaster() ) return;

    ptrDevicePtr = (MonteRay_CartesianGrid**) MONTERAYDEVICEALLOC(sizeof(MonteRay_CartesianGrid*), std::string("device - MonteRay_CartesianGrid::ptrDevicePtr") );

    pGridBins[0]->copyToGPU();
    pGridBins[1]->copyToGPU();
    pGridBins[2]->copyToGPU();

    std::unique_ptr<ptrCartesianGrid_result_t> ptrResult = std::unique_ptr<ptrCartesianGrid_result_t>( new ptrCartesianGrid_result_t() );
    ptrResult->copyToGPU();

    createDeviceInstance<<<1,1>>>( ptrDevicePtr, ptrResult->devicePtr, pGridBins[0]->devicePtr, pGridBins[1]->devicePtr, pGridBins[2]->devicePtr );
    cudaDeviceSynchronize();
    ptrResult->copyToCPU();
    devicePtr = ptrResult->v;

#endif
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
            return OUTSIDE_INDEX;
        }
    }

    if( debug ) printf("Debug: MonteRay_CartesianGrid::getIndex -- calling calcIndex\n");
    return calcIndex( indices );
}

CUDA_CALLABLE_MEMBER
gpuRayFloat_t
MonteRay_CartesianGrid::getVolume(unsigned index ) const {

    gpuRayFloat_t volume=1.0;
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
MonteRay_CartesianGrid::rayTrace(
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const GridBins_t::Position_t& particle_pos,
        const GridBins_t::Position_t& particle_dir,
        const gpuRayFloat_t distance,
        const bool outsideDistances ) const{

    if( debug ) printf( "Debug: MonteRay_CartesianGrid::rayTrace -- \n");

    int indices[3] = {0, 0, 0}; // current position indices in the grid, must be int because can be outside

    for( unsigned d=0; d<DIM; ++d){

        indices[d] = getDimIndex(d, particle_pos[d] );

        if( debug ) printf( "Debug: MonteRay_CartesianGrid::rayTrace -- dimension=%d, index=%d\n", d, indices[d]);

        planarCrossingDistance( d, threadID, rayInfo, *(pGridBins[d]), particle_pos[d], particle_dir[d], distance,indices[d]);

        if( debug ) printf( "Debug: MonteRay_CartesianGrid::rayTrace -- dimension=%d, number of planar crossings = %d\n", d, rayInfo.getCrossingSize(d,threadID) );

        // if outside and ray doesn't move inside then ray never enters the grid
        if( isIndexOutside(d,indices[d]) && rayInfo.getCrossingSize(d,threadID) == 0  ) {
            return;
        }
    }

    orderCrossings<3>( threadID, rayInfo, indices, distance, outsideDistances );

    if( debug ) printf( "Debug: MonteRay_CartesianGrid::rayTrace -- number of total crossings = %d\n", rayInfo.getRayCastSize(threadID) );
    return;
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CartesianGrid::crossingDistance(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const GridBins_t::Position_t& pos,
        const GridBins_t::Direction_t& dir,
        const gpuRayFloat_t distance ) const {

    crossingDistance( dim, threadID, rayInfo, pos[dim], dir[dim], distance);
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CartesianGrid::crossingDistance(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const gpuRayFloat_t pos,
        const gpuRayFloat_t dir,
        const gpuRayFloat_t distance ) const {

    if( debug ) printf( "Debug: MonteRay_CartesianGrid::crossingDistance( dim, threadID, rayInfo, float_t pos, float_t dir, float_t distance ) const \n");
    crossingDistance(dim, threadID, rayInfo, *(pGridBins[dim]), pos, dir, distance, false);
    return;
}

CUDA_CALLABLE_MEMBER
void
MonteRay_CartesianGrid::crossingDistance(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const GridBins_t& Bins,
        const gpuRayFloat_t pos,
        const gpuRayFloat_t dir,
        const gpuRayFloat_t distance,
        const bool equal_spacing) const {

    if( debug ) printf( "Debug: MonteRay_CartesianGrid::crossingDistance( dim, threadID, rayInfo, GridBins_t& Bins, float_t pos, float_t dir, float_t distance, bool equal_spacing) const \n");
    int index = Bins.getLinearIndex(pos);
    if( debug ) printf( "Debug: MonteRay_CartesianGrid::crossingDistance -- calling MonteRay_GridSystemInterface::planarCrossingDistance.\n");
    planarCrossingDistance( dim, threadID, rayInfo, Bins, pos, dir, distance, index);
    return;
}


} /* namespace MonteRay */
