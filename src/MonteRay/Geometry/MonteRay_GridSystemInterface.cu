/*
 * MonteRayGridSystemInterface.cc
 *
 *  Created on: Feb 2, 2018
 *      Author: jsweezy
 */

#include "MonteRay_GridSystemInterface.t.hh"
#include <float.h>

#include "RayWorkInfo.hh"

namespace MonteRay {

CUDA_CALLABLE_MEMBER
void
singleDimRayTraceMap_t::add( const int cell, const gpuRayFloat_t dist) {
    MONTERAY_ASSERT( N < MAXNUMVERTICES-1);
    CellId[N] = cell;
    distance[N] = dist;
    ++N;
}

// for conversion of old tests
CUDA_CALLABLE_MEMBER
singleDimRayTraceMap_t::singleDimRayTraceMap_t(RayWorkInfo& rayInfo, const unsigned threadID, int dim){

    if( dim < 0 ) {
        unsigned raySize = rayInfo.getRayCastSize(threadID);
        for( unsigned i=0; i<raySize; ++i ) {
            add( rayInfo.getRayCastCell(threadID,i),  rayInfo.getRayCastDist(threadID,i));
        }
        return;
    }

    for( unsigned i=0; i<rayInfo.getCrossingSize(dim,threadID); ++i ) {
        add( rayInfo.getCrossingCell(dim,threadID,i),  rayInfo.getCrossingDist(dim,threadID,i));
    }
}

CUDA_CALLABLE_MEMBER
void
rayTraceList_t::add( const unsigned cell, const gpuRayFloat_t dist) {
    MONTERAY_ASSERT( N < 2*MAXNUMVERTICES-1);
    CellId[N] = cell;
    distance[N] = dist;
    ++N;
}

// for conversion of old tests
CUDA_CALLABLE_MEMBER
rayTraceList_t::rayTraceList_t(RayWorkInfo& rayInfo, const unsigned threadID, int dim){

    if( dim < 0 ) {
        for( unsigned i=0; i<rayInfo.getRayCastSize(threadID); ++i ) {
            add( rayInfo.getRayCastCell(threadID,i),  rayInfo.getRayCastDist(threadID,i));
        }
        return;
    }

    for( unsigned i=0; i<rayInfo.getCrossingSize(dim,threadID); ++i ) {
        add( rayInfo.getCrossingCell(dim,threadID,i),  rayInfo.getCrossingDist(dim,threadID,i));
    }
}

template<unsigned NUMDIM>
CUDA_CALLABLE_MEMBER
void
MonteRay_GridSystemInterface::orderCrossings(
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        int indices[],
        const gpuRayFloat_t distance,
        const bool outsideDistances ) const {

    // Order the distance crossings to provide a rayTrace

#ifndef NDEBUG
    const bool debug = false;

    if( debug ) {
        printf( "Debug: *************************************************************\n");
        printf( "Debug: Starting GridSystemInterface::orderCrossings\n");
    }

    if( debug )  {
        for( unsigned d = 0; d<NUMDIM; ++d) {
            printf( "Debug: GridSystemInterface::orderCrossings -- dim=%d\n",d);
            for( unsigned i = 0; i< rayInfo.getCrossingSize(d,threadID); ++i) {
                printf( "Debug: ----------------------------------- -- distances[%d].id[%d]=%d, distances[%d].dist[%d]=%f\n", d,i, rayInfo.getCrossingCell(d,threadID,i), d,i, rayInfo.getCrossingDist(d,threadID,i));
            }
        }
    }
#endif

    unsigned start[NUMDIM]; // current location in the distance[i] vector
    unsigned   end[NUMDIM]; //    last location in the distance[i] vector

    unsigned maxNumCrossings = 0;
    for( unsigned i=0; i<NUMDIM; ++i){
        start[i] = 0;
        end[i] = rayInfo.getCrossingSize(i,threadID);
        maxNumCrossings += end[i];
    }

#ifndef NDEBUG
    if( debug ) printf( "Debug: GridSystemInterface::orderCrossings -- maxNumCrossings=%d\n",maxNumCrossings);
#endif

    gpuRayFloat_t minDistances[NUMDIM];

    bool outside;
    gpuRayFloat_t priorDistance = 0.0;

    for( unsigned i=0; i<maxNumCrossings; ++i){

        for( unsigned d = 0; d<NUMDIM; ++d) {
            if( start[d] < end[d] ) {
                minDistances[d] = rayInfo.getCrossingDist(d, threadID, start[d] );
            } else {
                minDistances[d] = inf;
            }
        }

#ifndef NDEBUG
        if( debug )  {
            for( unsigned d = 0; d<NUMDIM; ++d) {
                printf( "Debug: GridSystemInterface::orderCrossings -- dim=%u, minDistance[%u]=%f\n",d, d, minDistances[d]);
            }
        }
#endif

        unsigned minDim = 0;
        gpuRayFloat_t minDist = minDistances[0];
        for( unsigned i = 1; i<NUMDIM; ++i){
            if( minDistances[i] < minDist ) {
                minDim = i;
                minDist = minDistances[i];
            }
        }

#ifndef NDEBUG
        if( debug ) printf( "Debug: GridSystemInterface::orderCrossings -- minDim=%d\n",minDim);
        if( debug ) printf( "Debug: GridSystemInterface::orderCrossings -- minDist=%f\n",minDist);
#endif

        indices[minDim] = rayInfo.getCrossingCell(minDim, threadID, start[minDim] );

        // test for outside of the grid
        outside = isOutside( indices );

#ifndef NDEBUG
        if( debug ) {
            if( outside )  printf( "Debug: ray is outside \n" );
            if( !outside ) printf( "Debug: ray is inside \n" );
        }
#endif

        gpuRayFloat_t currentDistance = rayInfo.getCrossingDist(minDim, threadID, start[minDim] );

        if( !outside || outsideDistances ) {
            gpuRayFloat_t deltaDistance = currentDistance - priorDistance;

            MONTERAY_ASSERT_MSG( ( deltaDistance >= 0.0 ),
                    "ERROR:  MONTERAY -- MonteRay_GridSystemInterface::orderCrossings, delta distance is negative");

            unsigned global_index;
            if( !outside ) {
                global_index = calcIndex( indices );
            } else {
                global_index = MonteRay_GridSystemInterface::OUTSIDE_GRID;
            }
            rayInfo.addRayCastCell( threadID, global_index, deltaDistance );

#ifndef NDEBUG
            if( debug ) {
                printf( "Debug: ****************** \n" );
                printf( "Debug:  Entry Num    = %d\n", rayInfo.getRayCastSize(threadID) );
                printf( "Debug:     index[0]  = %d\n", indices[0] );
                printf( "Debug:     index[1]  = %d\n", indices[1] );
                printf( "Debug:     index[2]  = %d\n", indices[2] );
                printf( "Debug:     distance  = %f\n", deltaDistance );
            }
#endif

        }

        if( currentDistance >= distance ) {
            break;
        }

#ifndef NDEBUG
        if( debug ) {
            if( start[minDim]+1 >= rayInfo.getCrossingSize(minDim,threadID) ) {
                printf( "Debug: Error - start[minDim]+1 >= distances[minDim].size() \n");
                printf( "Debug:                   minDim = %d\n", minDim );
                printf( "Debug:          start[minDim]+1 = %d\n", start[minDim]+1 );
                printf( "Debug: distances[minDim].size() = %d\n", rayInfo.getCrossingSize(minDim,threadID) );
            }
        }
#endif

        MONTERAY_ASSERT( minDim < NUMDIM );
        MONTERAY_ASSERT( start[minDim]+1 < rayInfo.getCrossingSize(minDim,threadID) );

        indices[minDim] = rayInfo.getCrossingCell(minDim, threadID, start[minDim]+1 );

        if( ! outside ) {
            if( isIndexOutside(minDim, indices[minDim] ) ) {
                // ray has moved outside of grid
                break;
            }
        }

        ++start[minDim];
        priorDistance = currentDistance;
    }

    return;
}

template
CUDA_CALLABLE_MEMBER
void
MonteRay_GridSystemInterface::orderCrossings<1U>(
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        int indices[],
        const gpuRayFloat_t distance,
        const bool outsideDistances ) const;

template
CUDA_CALLABLE_MEMBER
void
MonteRay_GridSystemInterface::orderCrossings<2U>(
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        int indices[],
        const gpuRayFloat_t distance,
        const bool outsideDistances ) const;

template
CUDA_CALLABLE_MEMBER
void
MonteRay_GridSystemInterface::orderCrossings<3U>(
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        int indices[],
        const gpuRayFloat_t distance,
        const bool outsideDistances ) const;

CUDA_CALLABLE_MEMBER
void MonteRay_GridSystemInterface::planarCrossingDistance(
        const unsigned dim,
        const unsigned threadID,
        RayWorkInfo& rayInfo,
        const GridBins_t& Bins,
        const gpuRayFloat_t pos,
        const gpuRayFloat_t dir,
        const gpuRayFloat_t distance,
        const int start_index) const {

#ifndef NDEBUG
    const bool debug = false;

    if( debug ) printf( "Debug: MonteRay_GridSystemInterface::planarCrossingDistance --- \n" );
#endif

    //  constexpr gpuRayFloat_t epsilon = std::numeric_limits<gpuRayFloat_t>::epsilon();
#ifdef __CUDACC__
    if( abs(dir) <= FLT_EPSILON ) { return; }
#else
    if( std::abs(dir) <= FLT_EPSILON ) { return; }
#endif

#ifndef NDEBUG
    if( debug ) printf( "Debug: MonteRay_GridSystemInterface::planarCrossingDistance  -- Bins=%p \n", &Bins );
#endif

    if( start_index < 0 ) {
        if( dir < 0.0 ) {
            return;
        }
    }

    int nBins = Bins.getNumBins();

#ifndef NDEBUG
    if( debug ) printf( "Debug: MonteRay_GridSystemInterface::planarCrossingDistance - nBins=%d\n", nBins );
#endif

    if( start_index >= nBins ) {
        if( dir > 0.0 ) {
            return;
        }
    }

#ifdef __CUDA_ARCH__
    unsigned offset = int(signbit(-dir));
#else
    unsigned offset = int(std::signbit(-dir));
#endif

#ifndef NDEBUG
    if( debug ) printf( "Debug: MonteRay_GridSystemInterface::planarCrossingDistance - offset=%d\n", offset );
#endif

    int end_index = offset*(nBins-1);;

#ifdef __CUDA_ARCH__
    int dirIncrement = copysignf( 1, dir );
#else
    int dirIncrement = std::copysign( 1, dir );
#endif

#ifdef __CUDACC__
    unsigned num_indices = abs(end_index - start_index ) + 1;
#else
    unsigned num_indices = std::abs(end_index - start_index ) + 1;
#endif

    int current_index = start_index;

    // Calculate boundary crossing distances
    gpuRayFloat_t invDir = 1/dir;
    bool rayTerminated = false;
    for( int i = 0; i < num_indices ; ++i ) {

        //MONTERAY_ASSERT( (current_index + offset) >= 0 );
        MONTERAY_ASSERT( (current_index + offset) < Bins.getNumBins()+1 );

        gpuRayFloat_t minDistance = ( Bins.vertices[current_index + offset] - pos) * invDir;

        //if( rayDistance == inf ) {
        //    // ray doesn't cross plane
        //    break;
        //}

        if( minDistance >= distance ) {
            rayInfo.addCrossingCell(dim, threadID, current_index, distance );
            rayTerminated = true;
            break;
        }

        if( minDistance < 0.0 ) {
            minDistance = 0.0;
        }
        rayInfo.addCrossingCell(dim, threadID, current_index, minDistance );

        current_index += dirIncrement;
    }

    if( !rayTerminated ) {
        // finish with distance into area outside
        rayInfo.addCrossingCell(dim, threadID, current_index, distance );
        rayTerminated = true;
    }

    return;
}

} /* namespace MonteRay */
