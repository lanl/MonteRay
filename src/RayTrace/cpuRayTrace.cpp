#include <iostream>
#include <math.h>

#include "cpuRayTrace.h"
#include "GridBins.h"

unsigned rayTrace(const GridBins* const grid, int* global_indices, float_t* distances, const Position_t& pos, const Position_t& dir, float_t distance,  bool outsideDistances) {
	const bool debug = false;

    int current_indices[3] = {0, 0, 0}; // current position indices in the grid, must be int because can be outside

    if( debug ){
    	std::cout << "GridBins::rayTrace --------------------------------\n";
    }

    int cells[3][MAXNUMVERTICES];
    float_t crossingDistances[3][MAXNUMVERTICES];
    unsigned numCrossings[3];

    for( unsigned i=0; i<3; ++i){
    	current_indices[i] = getDimIndex(grid, i, pos[i] );

    	numCrossings[i] = calcCrossings( grid->vertices + grid->offset[i], grid->num[i]+1, cells[i], crossingDistances[i], pos[i], dir[i], distance, current_indices[i]);

    	if( debug ){
    		std::cout << "GridBins::rayTrace -- current_indices[i]=" << current_indices[i] << "\n";
    		std::cout << "GridBins::rayTrace -- numCrossings[i]=" << numCrossings[i] << "\n";
    	}

        // if outside and ray doesn't move inside then ray never enters the grid
        if( isIndexOutside(grid, i,current_indices[i]) && numCrossings[i] == 0  ) {
            return 0U;
        }
    }

    if( debug ){
    	std::cout << "GridBins::rayTrace -- numCrossings[0]=" << numCrossings[0] << "\n";
    	std::cout << "GridBins::rayTrace -- numCrossings[1]=" << numCrossings[1] << "\n";
    	std::cout << "GridBins::rayTrace -- numCrossings[2]=" << numCrossings[2] << "\n";
    }

    return orderCrossings(grid, global_indices, distances, MAXNUMVERTICES, cells[0], crossingDistances[0], numCrossings, current_indices, distance, outsideDistances);
}

unsigned calcCrossings(const float_t* const vertices, unsigned nVertices, int* cells, float_t* distances, float_t pos, float_t dir, float_t distance, int index ){
	const bool debug = false;

	unsigned nDistances = 0;

	if( debug ) {
		std::cout << "GridBins::calcCrossings --------------------------------\n";
		std::cout << "calcCrossings -- vertices[0]=" << vertices[0] << "\n";
		std::cout << "calcCrossings -- vertices[nVertices-1]=" << vertices[nVertices-1] << "\n";
		std::cout << "calcCrossings -- pos=" << pos << "\n";
		std::cout << "calcCrossings -- dir=" << dir << "\n";
	}

    if( abs(dir) <= global::epsilon ) {
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

    unsigned offset = 0;
    if( dir > 0.0f ) {
    	offset = 1;
    }
//    unsigned offset = (unsigned) signbit(-dir);
    int end_index = offset*(nBins-1);;

    int dirIncrement = copysign( 1.0f, dir );

    unsigned num_indices = abs(end_index - start_index ) + 1;

    int current_index = start_index;

    // Calculate boundary crossing distances
    float_t invDir = 1/dir;
    bool rayTerminated = false;
    for( int i = 0; i < num_indices ; ++i ) {

//        BOOST_ASSERT( (current_index + offset) >= 0 );
//        BOOST_ASSERT( (current_index + offset) < nBins+1 );

        float_t minDistance = ( vertices[current_index + offset] - pos) * invDir;

        if( debug ) {
        	std::cout << " calcCrossings -- current_index=" << current_index << "\n";
        	std::cout << " calcCrossings --        offset=" << offset << "\n";
        	std::cout << " calcCrossings -- vertices[current_index + offset]=" << vertices[current_index + offset] << "\n";
        }

        //if( rayDistance == inf ) {
        //    // ray doesn't cross plane
        //    break;
        //}

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

    if( debug ) {
    	for( unsigned i=0; i<nDistances; ++i){
    		std::cout << " calcCrossings -- i=" << i << " cell index=" << cells[i] << " distance=" << distances[i] << "\n";
    	}
    	std::cout << "-----------------------------------------------------------------------\n";
    }

    return nDistances;
}

unsigned orderCrossings(const GridBins* const grid, int* global_indices, float_t* distances, unsigned num, const int* const cells, const float_t* const crossingDistances, unsigned* numCrossings, int* indices, float_t distance, bool outsideDistances ){
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
    	float_t minimumDistance = global::inf;
        for( unsigned j = 0; j<3; ++j) {
            if( start[j] < end[j] ) {
            	minDistances[j] = *((crossingDistances+j*num)+start[j]);
            	if( minDistances[j] < minimumDistance ) {
            		minimumDistance = minDistances[j];
            		minDim = j;
            	}
            } else {
                minDistances[j] = global::inf;
            }
        }

        indices[minDim] =  *((cells+minDim*num) + start[minDim]);
        if( debug ) {
        	std::cout << "Debug: minDim=" << minDim << " index=" << indices[minDim] << " minimumDistance=" << minimumDistance << "\n";
        }

        // test for outside of the grid
        outside = isOutside(grid, indices );

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
                    global_index = calcIndex(grid, indices );
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
            if( isIndexOutside(grid, minDim, indices[minDim] ) ) {
                // ray has moved outside of grid
                break;
            }
        }

        ++start[minDim];
        priorDistance = currentDistance;
    }

    return numRayCrossings;
}
