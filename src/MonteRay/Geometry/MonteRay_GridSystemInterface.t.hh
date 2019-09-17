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
#include "RayWorkInfo.hh"

namespace MonteRay {

template<bool OUTWARD>
CUDA_CALLABLE_MEMBER
bool
MonteRay_GridSystemInterface::radialCrossingDistanceSingleDirection(
            const unsigned dim,
            const unsigned threadID,
            RayWorkInfo& rayInfo,
            const GridBins_t& Bins,
            const gpuRayFloat_t particle_R2,
            const gpuRayFloat_t A,
            const gpuRayFloat_t B,
            const gpuRayFloat_t distance,
            int index) const {

#ifndef __CUDA_ARCH__
    const gpuRayFloat_t Epsilon = 100.0 * std::numeric_limits<gpuRayFloat_t>::epsilon();
#else
#if RAY_DOUBLEPRECISION < 1
    const gpuRayFloat_t Epsilon = 100.0 * FLT_EPSILON;
#else
    const gpuRayFloat_t Epsilon = 100.0 * DBL_EPSILON;
#endif
#endif

    const bool outward = OUTWARD;

#ifndef NDEBUG
    const bool debug = false;
#endif

#ifndef NDEBUG
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
        rayInfo.addCrossingCell(dim, threadID, Bins.getNumBins(), distance);
        return true;
    }

    // if at lowest index and moving inward return
    if( !outward && index == 0 ) {
        return false;
    }

    unsigned MAX_DISTANCE_DIM = 2; // place max_distance in third dimension of RayWorkInfo

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

#ifndef NDEBUG
        if( debug ){
            printf("Debug: minDistance=%f, maxDistance=%f\n", minDistance, maxDistance);
        }
#endif

        if( minDistance == inf ) {
            // ray doesn't cross cylinder, terminate search
            break;
        }

        if( minDistance >= distance ) {
            rayInfo.addCrossingCell(dim, threadID, current_index, distance);
            rayTerminated = true;
            break;
        }

        if( minDistance > 0.0 ) {
            rayInfo.addCrossingCell(dim, threadID, current_index, minDistance);
        }

        if( ! outward ) {
            // rays directed inward can have two crossings
            if( maxDistance > 0.0 && maxDistance < inf) {
                rayInfo.addCrossingCell(MAX_DISTANCE_DIM, threadID, current_index-1, maxDistance);
            }
        }

        current_index += dirIncrement;
    }

    if( ! outward && ! rayTerminated ) {
        for( int i= rayInfo.getCrossingSize(MAX_DISTANCE_DIM,threadID)-1; i>=0; --i ){

            auto id_max = rayInfo.getCrossingCell(MAX_DISTANCE_DIM,threadID,i);
            auto dist_max = rayInfo.getCrossingDist(MAX_DISTANCE_DIM,threadID,i);
            if( dist_max > distance ) {
                rayInfo.addCrossingCell(dim, threadID, id_max, distance);
                rayTerminated = true;
                break;
            }
            rayInfo.addCrossingCell(dim, threadID, id_max, dist_max);
        }

    }
    if( outward && !rayTerminated ) {
        // finish with distance into area outside of largest radius
        rayInfo.addCrossingCell(dim, threadID, Bins.getNumBins(), distance);
        rayTerminated = true;
    }
    return rayTerminated;
}

} // end namespace

#endif /* MONTERAY_GRIDSYSTEMINTERFACE_T_HH_ */
