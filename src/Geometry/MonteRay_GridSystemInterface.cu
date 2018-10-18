/*
 * MonteRayGridSystemInterface.cc
 *
 *  Created on: Feb 2, 2018
 *      Author: jsweezy
 */

#include "MonteRay_GridSystemInterface.hh"

#include "MonteRay_QuadraticRootFinder.hh"
#include "GPUErrorCheck.hh"
#include <float.h>

namespace MonteRay {

CUDA_CALLABLE_MEMBER
void
singleDimRayTraceMap_t::add( int cell, gpuRayFloat_t dist) {
    MONTERAY_ASSERT( N < MAXNUMVERTICES-1);
    CellId[N] = cell;
    distance[N] = dist;
    ++N;
}

CUDA_CALLABLE_MEMBER
void
rayTraceList_t::add( unsigned cell, gpuRayFloat_t dist) {
    MONTERAY_ASSERT( N < MAXNUMVERTICES-1);
    CellId[N] = cell;
    distance[N] = dist;
    ++N;
}


CUDA_CALLABLE_MEMBER
void
MonteRay_GridSystemInterface::orderCrossings(rayTraceList_t& rayTraceList, const multiDimRayTraceMap_t& distances, int indices[], gpuRayFloat_t distance, bool outsideDistances ) const {
    // Order the distance crossings to provide a rayTrace

    const bool debug = false;

    if( debug ) {
        printf( "Debug: *************************************************************\n");
        printf( "Debug: Starting GridSystemInterface::orderCrossings\n");
    }

    if( debug )  {
        for( unsigned d = 0; d<DIM; ++d) {
            printf( "Debug: GridSystemInterface::orderCrossings -- dim=%d\n",d);
            for( unsigned i = 0; i<distances[d].size(); ++i) {
                printf( "Debug: ----------------------------------- -- distances[%d].id[%d]=%d, distances[%d].dist[%d]=%f\n", d,i, distances[d].id(i), d,i, distances[d].dist(i));
            }
        }
    }

    unsigned   end[3] = {0, 0, 0}; //    last location in the distance[i] vector

    unsigned maxNumCrossings = 0;
    for( unsigned i=0; i<DIM; ++i){
        end[i] = distances[i].size();
        maxNumCrossings += end[i];
    }

    if( debug ) printf( "Debug: GridSystemInterface::orderCrossings -- maxNumCrossings=%d\n",maxNumCrossings);

    // reset raylist
    rayTraceList.reset();

    gpuRayFloat_t minDistances[MAXDIM];

    bool outside;
    gpuRayFloat_t priorDistance = 0.0;
    unsigned start[3] = {0, 0, 0}; // current location in the distance[i] vector
    for( unsigned i=0; i<maxNumCrossings; ++i){

        for( unsigned d = 0; d<DIM; ++d) {
            if( start[d] < end[d] ) {
                minDistances[d] = distances[d].dist( start[d] );
            } else {
                minDistances[d] = inf;
            }
        }

        if( debug )  {
            for( unsigned d = 0; d<DIM; ++d) {
                printf( "Debug: GridSystemInterface::orderCrossings -- dim=%u, minDistance[%u]=%f\n",d, d, minDistances[d]);
            }
        }

        //unsigned minDim = std::distance(minDistances, std::min_element(minDistances,minDistances+DIM) );
        unsigned minDim = 0;
        gpuRayFloat_t minDist = minDistances[0];
        for( unsigned i = 1; i<DIM; ++i){
            if( minDistances[i] < minDist ) {
                minDim = i;
                minDist = minDistances[i];
            }
        }

        if( debug ) printf( "Debug: GridSystemInterface::orderCrossings -- minDim=%d\n",minDim);
        if( debug ) printf( "Debug: GridSystemInterface::orderCrossings -- minDist=%f\n",minDist);

        //indices[minDim] = distances[minDim][start[minDim]].first;
        indices[minDim] = distances[minDim].id( start[minDim] );

        // test for outside of the grid
        outside = isOutside( indices );

        if( debug ) {
            if( outside )  printf( "Debug: ray is outside \n" );
            if( !outside ) printf( "Debug: ray is inside \n" );
        }

        //gpuRayFloat_t currentDistance = distances[minDim][start[minDim]].second;
        gpuRayFloat_t currentDistance = distances[minDim].dist( start[minDim] );

        if( !outside || outsideDistances ) {
            gpuRayFloat_t deltaDistance = currentDistance - priorDistance;

            unsigned global_index;
            if( !outside ) {
                global_index = calcIndex( indices );
            } else {
                global_index = MonteRay_GridSystemInterface::OUTSIDE_GRID;
            }
            rayTraceList.add( global_index, deltaDistance );

            if( debug ) {
                printf( "Debug: ****************** \n" );
                printf( "Debug:  Entry Num    = %d\n", rayTraceList.size() );
                printf( "Debug:     index[0]  = %d\n", indices[0] );
                printf( "Debug:     index[1]  = %d\n", indices[1] );
                printf( "Debug:     index[2]  = %d\n", indices[2] );
                printf( "Debug:     distance  = %f\n", deltaDistance );
            }
        }

        if( currentDistance >= distance ) {
            break;
        }

        if( debug ) {
            if( start[minDim]+1 >= distances[minDim].size() ) {
                printf( "Debug: Error - start[minDim]+1 >= distances[minDim].size() \n");
                printf( "Debug:                   minDim = %d\n", minDim );
                printf( "Debug:          start[minDim]+1 = %d\n", start[minDim]+1 );
                printf( "Debug: distances[minDim].size() = %d\n", distances[minDim].size() );
            }
        }

        MONTERAY_ASSERT( minDim < 3 );
        //MONTERAY_ASSERT( minDim < distances.size() );
        MONTERAY_ASSERT( start[minDim]+1 < distances[minDim].size() );

        //indices[minDim] = distances[minDim][start[minDim]+1].first;
        indices[minDim] = distances[minDim].id( start[minDim]+1 );

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

CUDA_CALLABLE_MEMBER
void
MonteRay_GridSystemInterface::planarCrossingDistance(singleDimRayTraceMap_t& distances, const GridBins_t& Bins, gpuRayFloat_t pos, gpuRayFloat_t dir, gpuRayFloat_t distance, int index) const {
    const bool debug = false;
    if( debug ) printf( "Debug: MonteRay_GridSystemInterface::planarCrossingDistance --- \n" );

    //	constexpr gpuRayFloat_t epsilon = std::numeric_limits<gpuRayFloat_t>::epsilon();
#ifdef __CUDACC__
    if( abs(dir) <= FLT_EPSILON ) { return; }
#else
    if( std::abs(dir) <= FLT_EPSILON ) { return; }
#endif


    if( debug ) printf( "Debug: MonteRay_GridSystemInterface::planarCrossingDistance  -- Bins=%p \n", &Bins );

    int start_index = index;
    int cell_index = start_index;

    if( start_index < 0 ) {
        if( dir < 0.0 ) {
            return;
        }
    }

    int nBins = Bins.getNumBins();
    if( debug ) printf( "Debug: MonteRay_GridSystemInterface::planarCrossingDistance - nBins=%d\n", nBins );
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
    if( debug ) printf( "Debug: MonteRay_GridSystemInterface::planarCrossingDistance - offset=%d\n", offset );
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
            distances.add( cell_index, distance);
            rayTerminated = true;
            break;
        }

        distances.add( cell_index, minDistance);

        current_index += dirIncrement;
        cell_index = current_index;
    }

    if( !rayTerminated ) {
        // finish with distance into area outside
        distances.add( cell_index, distance);
        rayTerminated = true;
    }

    return;
}

CUDA_CALLABLE_MEMBER
bool
MonteRay_GridSystemInterface::radialCrossingDistanceSingleDirection( singleDimRayTraceMap_t& distances, const GridBins_t& Bins, gpuRayFloat_t particle_R2, gpuRayFloat_t A, gpuRayFloat_t B, gpuRayFloat_t distance, int index, bool outward ) const {
    const bool debug = false;
    if( debug ){
        printf("Debug: MonteRay_GridSystemInterface::radialCrossingDistanceSingleDirection -- \n");
    }

    // If outside and moving out then return
    if( outward && index >= Bins.getNumBins() ) {
        //distances.push_back( std::make_pair( Bins.getNumBins(), distance)  );
        distances.add(Bins.getNumBins(), distance);
        return true;
    }

    // if at lowest index and moving inward return
    if( !outward && index == 0 ) {
        return false;
    }

    singleDimRayTraceMap_t max_distances;

    int start_index = index;
    int cell_index = start_index;

    int dirIncrement = -1;
    int end_index = 1;
    int offset = -1;
    if( outward ) {
        offset = 0;
        dirIncrement = 1;
        end_index = Bins.getNumBins()-1;
    }

#ifdef __CUDACC__
    unsigned num_indices = abs(end_index - start_index ) + 1;
#else
    unsigned num_indices = std::abs(end_index - start_index ) + 1;
#endif
    //distances.reserve( num_indices+5 );

    int current_index = start_index;

    // Calculate boundary crossing distances
    bool rayTerminated = false;
    for( int i = 0; i < num_indices ; ++i ) {

        MONTERAY_ASSERT( (current_index + offset) >= 0 );
        MONTERAY_ASSERT( (current_index + offset) < Bins.getNumBins() );

        gpuRayFloat_t RadiusSq = Bins.verticesSq[current_index + offset ];
        gpuRayFloat_t C = particle_R2 - RadiusSq;

        Roots rayDistances = FindPositiveRoots(A,B,C);
        gpuRayFloat_t minDistance;
        gpuRayFloat_t maxDistance;
        if( rayDistances.R1 < rayDistances.R2 ) {
            minDistance = rayDistances.R1;
            maxDistance = rayDistances.R2;
        } else {
            minDistance = rayDistances.R2;
            maxDistance = rayDistances.R1;
        }
        if( debug ){
            printf("Debug: minDistance=%f, maxDistance=%f\n", minDistance, maxDistance);
        }

        if( minDistance == inf ) {
            // ray doesn't cross cylinder, terminate search
            break;
        }

        if( minDistance >= distance ) {
            //distances.push_back( std::make_pair( cell_index, distance)  );
            distances.add( cell_index, distance );
            rayTerminated = true;
            break;
        }

        if( minDistance > 0.0 ) {
            //distances.push_back( std::make_pair( cell_index, minDistance)  );
            distances.add( cell_index, minDistance );
        }

        if( ! outward ) {
            // rays directed inward can have two crossings
            if( maxDistance > 0.0 && maxDistance < inf) {
                //max_distances.push_back( std::make_pair( cell_index-1, maxDistance)  );
                max_distances.add( cell_index-1, maxDistance );
            }
        }

        current_index += dirIncrement;
        cell_index = current_index;
    }

    if( ! outward && ! rayTerminated ) {
        for( int i=max_distances.size()-1; i>=0; --i ){
            auto id_max = max_distances.id(i);
            auto dist_max = max_distances.dist(i);
            if( dist_max > distance ) {
                distances.add( id_max, distance );
                rayTerminated = true;
                break;
            }
            distances.add( id_max, dist_max );
        }

    }
    if( outward && !rayTerminated ) {
        // finish with distance into area outside of largest radius
        //        distances.push_back( std::make_pair( Bins.getNumBins(), distance)  );
        distances.add( Bins.getNumBins(), distance);
        rayTerminated = true;
    }
    return rayTerminated;

}

} /* namespace MonteRay */
