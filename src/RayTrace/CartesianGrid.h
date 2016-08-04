#ifndef CARTESIANGRID_H_
#define CARTESIANGRID_H_

#include <stdio.h>        /* perror */
#include <errno.h>        /* errno */
#include <stdlib.h>
#include <limits.h>
#include <iostream>

#include "global.h"
#include "Vector3D.h"
#include "cpuCrossingDistance.h"

namespace MonteRay{

#define MAXNUMVERTICES 1001

#define X 0
#define Y 1
#define Z 2

class CartesianGrid {
public:
	typedef global::float_t float_t;
	typedef Vector3D Position_t;


	CartesianGrid(){
		offset[0] = 0;
		offset[1] = MAXNUMVERTICES;
		offset[2] = MAXNUMVERTICES*2;
		num[0] = 0;
		num[1] = 0;
		num[2] = 0;
	};
	~CartesianGrid(){};

	void setVertices( unsigned dim, float_t min, float max, unsigned numBins ) {
		delta[dim] = (max - min) / numBins;
		num[dim] = numBins;

		if( numBins+1 > MAXNUMVERTICES ) {
			perror("CartesianGrid::setVertices -- exceeding max number of vertices.");
			exit(1);
		}
		vertices[offset[dim]] = min;

		float_t location;
		for( unsigned i = 1; i<numBins+1; ++i) {
			vertices[i+offset[dim]] = vertices[i-1 + offset[dim]] + delta[dim];
		}
	}

	void finalize(void) {
		for( unsigned dim = 0; dim < 3; ++dim) {
			if( num[dim] == 0 ) {
				perror("CartesianGrid::finalize -- vertices not set.");
			    exit(1);
			}
		}

		// move y data
		unsigned new_offset = num[0]+1;
		for( unsigned i = 0; i < num[1]+1; ++i) {
			vertices[i + new_offset] = vertices[i+offset[1]];
			vertices[i+offset[1]] = -1.0;
		}
		offset[1] = num[0]+1;

		// move z data
		new_offset = num[0]+num[1]+2;
		for( unsigned i = 0; i < num[2]+1; ++i) {
			vertices[i + new_offset] = vertices[i+offset[2]];
			vertices[i+offset[2]] = -1.0;
		}
		offset[2] = num[0]+1 + num[1]+1;
	}

	unsigned calcIndex( const int* const indices ) const {
	    return indices[0] + indices[1]*getNumBins(0) + indices[2]*getNumBins(0)*getNumBins(1);
	}

	void calcIJK( unsigned index, unsigned* indices ) const {
	    indices[0] = 0; indices[1] = 0; indices[2] = 0;

	    unsigned offsets[3];
	    offsets[0] = 1;
	    offsets[1] = getNumBins(0);
	    offsets[2] = getNumBins(0)*getNumBins(1);

	    for( int d = 2; d > -1; --d ) {
	        unsigned current_offset = offsets[ d ];
	        indices[d] = index / current_offset;
	        index -= indices[d] * current_offset;
	    }
	}

	unsigned getIndex( const Position_t& particle_pos) const{

	    int indices[3]= {0, 0, 0};
	    for( unsigned d = 0; d < 3; ++d ) {
	        indices[d] = getDimIndex(d, particle_pos[d] );

	        // outside the grid
	        if( isIndexOutside(d, indices[d] ) ) { return UINT_MAX; }
	    }

	    return calcIndex( indices );
	}

	int getDimIndex(unsigned dim, float_t pos ) const {
	     // returns -1 for one neg side of mesh
	     // and number of bins on the pos side of the mesh
	     // need to call isIndexOutside(dim, grid, index) to check if the
	     // index is in the mesh

		int dim_index;
		if( pos <= min(dim) ) {
			dim_index = -1;
		} else if( pos >= max(dim)  ) {
			dim_index = getNumBins(dim);
		} else {
			dim_index = ( pos -  min(dim) ) / delta[dim];
		}
		return dim_index;
	}

	bool isOutside( const int* indices ) const {
	    for( unsigned d=0; d<3; ++d){
	       if( isIndexOutside(d, indices[d]) ) return true;
	    }
	    return false;
	}

	bool isIndexOutside( unsigned dim, int i) const { if( i < 0 ||  i >= getNumBins(dim) ) return true; return false; }

	unsigned getMaxNumVertices(void) const { return MAXNUMVERTICES; }

	unsigned getNumBins(unsigned dim) const { return num[dim]; }
	unsigned getNumVertices(unsigned dim) const { return num[dim]+1; }

	float_t min(unsigned dim) const { return vertices[offset[dim]]; }
	float_t max(unsigned dim) const { return vertices[offset[dim]+num[dim]]; }

	float_t getVertex( unsigned dim, unsigned index ) const { return vertices[offset[dim]+index]; }


	unsigned crossingDistance( unsigned dim, int* cells, float_t* distances, float_t pos, float_t dir, float_t distance ){
		int index = getDimIndex(dim, pos);
		return CrossingDistance::calc( vertices+offset[dim], num[dim] + 1, cells, distances, pos, dir, distance, index);
	}

	unsigned rayTrace( int* global_indices, float_t* distances, const Position_t& pos, const Position_t& dir, float_t distance,  bool outsideDistances) const{
	    int current_indices[3] = {0, 0, 0}; // current position indices in the grid, must be int because can be outside

//	    std::cout << "CartesianGrid::rayTrace --------------------------------\n";

	    int cells[3][MAXNUMVERTICES];
	    float_t crossingDistances[3][MAXNUMVERTICES];
	    unsigned numCrossings[3];

	    for( unsigned i=0; i<3; ++i){
	    	current_indices[i] = getDimIndex(i, pos[i] );
//	        std::cout << "CartesianGrid::rayTrace -- current_indices[i]=" << current_indices[i] << "\n";

	    	numCrossings[i] = CrossingDistance::calc( vertices+offset[i], num[i]+1, cells[i], crossingDistances[i], pos[i], dir[i], distance, current_indices[i]);

//	    	std::cout << "CartesianGrid::rayTrace -- numCrossings[i]=" << numCrossings[i] << "\n";

	        // if outside and ray doesn't move inside then ray never enters the grid
	        if( isIndexOutside(i,current_indices[i]) && numCrossings[i] == 0  ) {
	            return 0U;
	        }
	    }
//	    std::cout << "CartesianGrid::rayTrace -- numCrossings[0]=" << numCrossings[0] << "\n";
//	    std::cout << "CartesianGrid::rayTrace -- numCrossings[1]=" << numCrossings[1] << "\n";
//	    std::cout << "CartesianGrid::rayTrace -- numCrossings[2]=" << numCrossings[2] << "\n";

	    return CrossingDistance::orderCrossings(this, global_indices, distances, MAXNUMVERTICES, cells[0], crossingDistances[0], numCrossings, current_indices, distance, outsideDistances);
	}


	float_t vertices[MAXNUMVERTICES*3];

	unsigned num[3];
	unsigned offset[3];

	float_t delta[3];

};

}

#endif /* CARTESIANGRID_H_ */
