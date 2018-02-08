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
MonteRay_GridSystemInterface::orderCrossings(rayTraceList_t& rayTraceList, const multiDimRayTraceMap_t& distances, int indices[], gpuFloatType_t distance, bool outsideDistances ) const {
    // Order the distance crossings to provide a rayTrace

    const bool debug = false;

    if( debug ) {
        printf( "Debug: *************************************************************\n");
        printf( "Debug: Starting GridSystemInterface::orderCrossings\n");
    }

    unsigned   end[3] = {0, 0, 0}; //    last location in the distance[i] vector

    unsigned maxNumCrossings = 0;
    for( unsigned i=0; i<DIM; ++i){
        end[i] = distances[i].size();
        maxNumCrossings += end[i];
    }

    // reset raylist
    rayTraceList.reset();

    gpuFloatType_t minDistances[MAXDIM];

    bool outside;
    gpuFloatType_t priorDistance = 0.0;
    unsigned start[3] = {0, 0, 0}; // current location in the distance[i] vector
    for( unsigned i=0; i<maxNumCrossings; ++i){

        for( unsigned d = 0; d<DIM; ++d) {
            if( start[d] < end[d] ) {
                minDistances[d] = distances[d].dist( start[d] );
            } else {
                minDistances[d] = inf;
            }
        }

        //unsigned minDim = std::distance(minDistances, std::min_element(minDistances,minDistances+DIM) );
        unsigned minDim = 0;
        gpuFloatType_t minDist = minDistances[0];
        for( unsigned i = 1; i<DIM; ++i){
        	if( minDistances[i] < minDist ) {
        		minDim = i;
        		minDist = minDistances[i];
        	}
        }

        //indices[minDim] = distances[minDim][start[minDim]].first;
        indices[minDim] = distances[minDim].id( start[minDim] );

        // test for outside of the grid
        outside = isOutside( indices );

        if( debug ) {
            if( outside )  printf( "Debug: ray is outside \n" );
            if( !outside ) printf( "Debug: ray is inside \n" );
        }

        //gpuFloatType_t currentDistance = distances[minDim][start[minDim]].second;
        gpuFloatType_t currentDistance = distances[minDim].dist( start[minDim] );

        if( !outside || outsideDistances ) {
        	gpuFloatType_t deltaDistance = currentDistance - priorDistance;

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
MonteRay_GridSystemInterface::planarCrossingDistance(singleDimRayTraceMap_t& distances, const GridBins_t& Bins, gpuFloatType_t pos, gpuFloatType_t dir, gpuFloatType_t distance, int index) const {
//	constexpr gpuFloatType_t epsilon = std::numeric_limits<gpuFloatType_t>::epsilon();
    if( abs(dir) <= FLT_EPSILON ) { return; }

    int start_index = index;
    int cell_index = start_index;

    if( start_index < 0 ) {
        if( dir < 0.0 ) {
            return;
        }
    }

    int nBins = Bins.getNumBins();
    if( start_index >= nBins ) {
        if( dir > 0.0 ) {
            return;
        }
    }

    unsigned offset = int(std::signbit(-dir));
    int end_index = offset*(nBins-1);;

#ifdef __CUDA_ARCH__
    int dirIncrement = copysignf( 1, dir );
#else
    int dirIncrement = std::copysign( 1, dir );
#endif

    unsigned num_indices = std::abs(end_index - start_index ) + 1;

    int current_index = start_index;

    // Calculate boundary crossing distances
    gpuFloatType_t invDir = 1/dir;
    bool rayTerminated = false;
    for( int i = 0; i < num_indices ; ++i ) {

    	//MONTERAY_ASSERT( (current_index + offset) >= 0 );
    	MONTERAY_ASSERT( (current_index + offset) < Bins.getNumBins()+1 );

        gpuFloatType_t minDistance = ( Bins.vertices[current_index + offset] - pos) * invDir;

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
MonteRay_GridSystemInterface::radialCrossingDistanceSingleDirection( singleDimRayTraceMap_t& distances, const GridBins_t& Bins, gpuFloatType_t particle_R2, gpuFloatType_t A, gpuFloatType_t B, gpuFloatType_t distance, int index, bool outward ) const {

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

    unsigned num_indices = std::abs(end_index - start_index ) + 1;
    //distances.reserve( num_indices+5 );

    int current_index = start_index;

    // Calculate boundary crossing distances
    bool rayTerminated = false;
    for( int i = 0; i < num_indices ; ++i ) {

    	MONTERAY_ASSERT( (current_index + offset) >= 0 );
    	MONTERAY_ASSERT( (current_index + offset) < Bins.getNumBins() );

        gpuFloatType_t RadiusSq = Bins.verticesSq[current_index + offset ];
        gpuFloatType_t C = particle_R2 - RadiusSq;

        Roots rayDistances = FindPositiveRoots(A,B,C);
        gpuFloatType_t minDistance;
        gpuFloatType_t maxDistance;
        if( rayDistances.R1 < rayDistances.R2 ) {
            minDistance = rayDistances.R1;
            maxDistance = rayDistances.R2;
        } else {
            minDistance = rayDistances.R2;
            maxDistance = rayDistances.R1;
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
//        for( auto itr = max_distances.rbegin(); itr != max_distances.rend(); ++itr ) {
//             if( (*itr).second > distance ) {
//                 distances.push_back( std::make_pair( (*itr).first, distance)  );
//                 rayTerminated = true;
//                 break;
//             }
//             distances.push_back( *itr );
//         }

    	for( unsigned i=0; i<max_distances.size(); ++i ){
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
