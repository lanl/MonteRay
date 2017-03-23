#include "cpuCrossingDistance.hh"

#include <cmath>
#include <math.h>
#include <iostream>
#include <limits.h>
#include "CartesianGrid.hh"

namespace MonteRay{

unsigned CrossingDistance::calc(const float_t* const vertices, unsigned nVertices, int* cells, float_t* distances, float_t pos, float_t dir, float_t distance, int index ){
	const bool debug = false;

	unsigned nDistances = 0;

	if( debug ) {
		std::cout << "CrossingDistance::calc -- vertices[0]=" << vertices[0] << "\n";
		std::cout << "CrossingDistance::calc -- vertices[nVertices-1]=" << vertices[nVertices-1] << "\n";
		std::cout << "CrossingDistance::calc -- pos=" << pos << "\n";
		std::cout << "CrossingDistance::calc -- dir=" << dir << "\n";
	}

    if( std::abs(dir) <= MonteRay::epsilon ) {
    	return nDistances;
    }

    int start_index = index;
    int cell_index = start_index;

    if( start_index < 0 ) {
        if( dir < 0.0 ) {
            return nDistances;
        }
    }

    int nBins = nVertices - 1;
    if( start_index >= nBins ) {
        if( dir > 0.0 ) {
        	return nDistances;
        }
    }

    unsigned offset = int(std::signbit(-dir));
    int end_index = offset*(nBins-1);;

    int dirIncrement = copysign( 1, dir );

    unsigned num_indices = std::abs(end_index - start_index ) + 1;

    int current_index = start_index;

    double invDir = 1/dir;
    bool rayTerminated = false;

    // Calculate boundary crossing distances
    for( int i = 0; i < num_indices ; ++i ) {

#if DEBUG >= 1
    	if( (current_index + offset) < 0 ) {
    		throw std::runtime_error( "cpuCrosingDistance::calc :: index negative :: (current_index + offset) < 0 ");
    	}
    	if( (current_index + offset) >= nBins+1 ) {
    		throw std::runtime_error( "cpuCrosingDistance::calc :: index exceeds valid range :: (current_index + offset) >= nBins+1 ");
    	}
#endif

    	double minDistance = ( vertices[current_index + offset] - pos) * invDir;

    	if( minDistance >= distance ) {
    		cells[nDistances] = cell_index;
    		distances[nDistances] = distance;
    		++nDistances;
    		rayTerminated = true;
    		break;
    	}

    	cells[nDistances] = cell_index;
    	distances[nDistances] = minDistance;
    	++nDistances;

    	current_index += dirIncrement;
    	cell_index = current_index;
    }

    if( !rayTerminated ) {
        // finish with distance into area outside
    	cells[nDistances] = cell_index;
    	distances[nDistances] = distance;
    	++nDistances;
        rayTerminated = true;
    }

    return nDistances;
}



unsigned CrossingDistance::orderCrossings(const CartesianGrid* const grid, int* global_indices, float_t* distances, unsigned num, const int* const cells, const float_t* const crossingDistances, unsigned* numCrossings, int* indices, double distance, bool outsideDistances ){
    // Order the distance crossings to provide a rayTrace

    const bool debug = false;

    unsigned end[3] = {0, 0, 0}; //    last location in the distance[i] vector

    unsigned maxNumCrossings = 0;
    for( unsigned i=0; i<3; ++i){
        end[i] = numCrossings[i];
        maxNumCrossings += end[i];
    }

    if( debug ) {
    	for( unsigned i=0; i<3; ++i){
    		std::cout << "Debug: i=" << i << " numCrossings=" << numCrossings[i] << "\n";
    		for( unsigned j=0; j< numCrossings[i]; ++j ) {
    			std::cout << "Debug:       j=" << j << " index=" << *((cells+i*num) + j) << " distance=" << *((crossingDistances+i*num)+j) << "\n";
    		}
    	}
    }

    float_t minDistances[3];

    bool outside;
    float_t priorDistance = 0.0;
    unsigned start[3] = {0, 0, 0}; // current location in the distance[i] vector

    unsigned numRayCrossings = 0;
    for( unsigned i=0; i<maxNumCrossings; ++i){

    	unsigned minDim;
    	float_t minimumDistance = MonteRay::inf;
        for( unsigned j = 0; j<3; ++j) {
            if( start[j] < end[j] ) {
            	minDistances[j] = *((crossingDistances+j*num)+start[j]);
            	if( minDistances[j] < minimumDistance ) {
            		minimumDistance = minDistances[j];
            		minDim = j;
            	}
            } else {
                minDistances[j] = MonteRay::inf;
            }
        }

        indices[minDim] =  *((cells+minDim*num) + start[minDim]);
        if( debug ) {
        	std::cout << "Debug: minDim=" << minDim << " index=" << indices[minDim] << " minimumDistance=" << minimumDistance << "\n";
        }

        // test for outside of the grid
        outside = grid->isOutside( indices );

        if( debug ) {
            if( outside )  std::cout << "Debug: ray is outside " << std::endl;
            if( !outside ) std::cout << "Debug: ray is inside " << std::endl;
        }

        float_t currentDistance = minimumDistance;

        if( !outside || outsideDistances ) {
        	float_t deltaDistance = currentDistance - priorDistance;

            if( deltaDistance > 0.0  ) {
                unsigned global_index;
                if( !outside ) {
                    global_index = grid->calcIndex( indices );
                } else {
                    global_index = UINT_MAX;
                }
                global_indices[numRayCrossings] = global_index;
                distances[numRayCrossings] = deltaDistance;
                ++numRayCrossings;

                if( debug ) {
                    std::cout << "Debug: ******************" << std::endl;
                    std::cout << "Debug:  Entry Num   = " << numRayCrossings << std::endl;
                    std::cout << "Debug:     index[0]  = " << indices[0]  << std::endl;
                    std::cout << "Debug:     index[1]  = " << indices[1]  << std::endl;
                    std::cout << "Debug:     index[2]  = " << indices[2]  << std::endl;
                    std::cout << "Debug:     distance = " << deltaDistance  << std::endl;
                }
            }
        }

        if( currentDistance >= distance ) {
            break;
        }

        indices[minDim] = *((cells+minDim*num) + start[minDim]+1);

        if( ! outside ) {
            if( grid->isIndexOutside(minDim, indices[minDim] ) ) {
                // ray has moved outside of grid
                break;
            }
        }

        ++start[minDim];
        priorDistance = currentDistance;
    }

    return numRayCrossings;
}
}
