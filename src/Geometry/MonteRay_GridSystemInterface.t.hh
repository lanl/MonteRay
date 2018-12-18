#ifndef MONTERAY_GRIDSYSTEMINTERFACE_T_HH_
#define MONTERAY_GRIDSYSTEMINTERFACE_T_HH_

#include "MonteRay_GridSystemInterface.hh"

#include <cmath>
#include <limits>

#ifdef __CUDACC__
#include <float.h>
#include <math_constants.h>
#endif

#include "MonteRayTypes.hh"
#include "GPUErrorCheck.hh"
#include "MonteRay_QuadraticRootFinder.hh"

namespace MonteRay {

template<bool OUTWARD>
CUDA_CALLABLE_MEMBER
bool
MonteRay_GridSystemInterface::radialCrossingDistanceSingleDirection( singleDimRayTraceMap_t& distances, const GridBins_t& Bins, gpuRayFloat_t particle_R2, gpuRayFloat_t A, gpuRayFloat_t B, gpuRayFloat_t distance, int index ) const {
#ifndef __CUDA_ARCH__
    const gpuRayFloat_t Epsilon = 100.0 * std::numeric_limits<Float_t>::epsilon();
#else
#if RAY_DOUBLEPRECISION < 1
    const gpuRayFloat_t Epsilon = 100.0 * FLT_EPSILON;
#else
    const gpuRayFloat_t Epsilon = 100.0 * DBL_EPSILON;
#endif
#endif

    const bool outward = OUTWARD;

#ifdef DEBUG
    const bool debug = false;
#endif

#ifdef DEBUG
    if( debug ){
        printf("Debug: MonteRay_GridSystemInterface::radialCrossingDistanceSingleDirection -- \n");
    }
#endif

    // Test to see if very near the surface and directed outward.
    // If so skip the surface
#ifdef __CUDACC__
    if( outward and abs( sqrt(particle_R2) - Bins.vertices[ index ] ) < Epsilon ) {
#else
    if( outward and std::abs( std::sqrt(particle_R2) - Bins.vertices[ index ] ) < Epsilon ) {
#endif
        ++index;
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

#ifdef DEBUG
        if( debug ){
            printf("Debug: minDistance=%f, maxDistance=%f\n", minDistance, maxDistance);
        }
#endif

        if( minDistance == inf ) {
            // ray doesn't cross cylinder, terminate search
            break;
        }

        if( minDistance >= distance ) {
            //distances.push_back( std::make_pair( cell_index, distance)  );
            distances.add( current_index, distance );
            rayTerminated = true;
            break;
        }

        if( minDistance > 0.0 ) {
            //distances.push_back( std::make_pair( cell_index, minDistance)  );
            distances.add( current_index, minDistance );
        }

        if( ! outward ) {
            // rays directed inward can have two crossings
            if( maxDistance > 0.0 && maxDistance < inf) {
                //max_distances.push_back( std::make_pair( cell_index-1, maxDistance)  );
                max_distances.add( current_index-1, maxDistance );
            }
        }

        current_index += dirIncrement;
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

}

#endif /* MONTERAY_GRIDSYSTEMINTERFACE_T_HH_ */
