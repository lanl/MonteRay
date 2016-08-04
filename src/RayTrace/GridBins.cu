#include "GridBins.h"
#include "gpuGlobal.h"

namespace MonteRay{

void ctor(GridBins* grid){
	grid->offset[0] = 0;
	grid->offset[1] = MAXNUMVERTICES;
	grid->offset[2] = MAXNUMVERTICES*2;
	grid->num[0] = 0;
	grid->num[1] = 0;
	grid->num[2] = 0;
	grid->numXY = 0;
};

unsigned getNumVertices(const GridBins* const grid, unsigned dim) {
	return grid->num[dim] + 1;
}

unsigned getNumXY(const GridBins* const grid) {
	return grid->numXY;
}

unsigned getNumBins(const GridBins* const grid, unsigned dim) {
	return grid->num[dim];
}

unsigned getNumBins(const GridBins* const grid, unsigned dim, unsigned index) {
	return grid->num[dim];
}

float_t getVertex(const GridBins* const grid, unsigned dim, unsigned index ) {
	return grid->vertices[ grid->offset[dim] + index ];
}

float_t min(const GridBins* const grid, unsigned dim) {
	return grid->minMax[dim*2];
}

float_t max(const GridBins* const grid, unsigned dim) {
	return grid->minMax[dim*2+1];
}

void setVertices( GridBins* grid, unsigned dim, global::float_t min, float max, unsigned numBins ) {

	grid->minMax[dim*2] = min;
	grid->minMax[dim*2+1] = max;

	grid->delta[dim] = (max - min) / numBins;
	grid->num[dim] = numBins;

	if( numBins+1 > MAXNUMVERTICES ) {
		perror("GridBins::setVertices -- exceeding max number of vertices.");
		exit(1);
	}

	grid->vertices[ grid->offset[dim] ] = min;

	//global::float_t location;
	for( unsigned i = 1; i<numBins+1; ++i) {
		grid->vertices[ i+ grid->offset[dim]] = grid->vertices[i-1 + grid->offset[dim]] + grid->delta[dim];
	}
}

void finalize(GridBins* grid) {
	for( unsigned dim = 0; dim < 3; ++dim) {
		if( grid->num[dim] == 0 ) {
			perror("GridBins::finalize -- vertices not set.");
		    exit(1);
		}
	}

	// move y data
	unsigned new_offset = grid->num[0]+1;
	for( unsigned i = 0; i < grid->num[1]+1; ++i) {
		grid->vertices[i + new_offset] = grid->vertices[i+grid->offset[1]];
		grid->vertices[i+grid->offset[1]] = -1.0;
	}
	grid->offset[1] = grid->num[0]+1;

	// move z data
	new_offset = grid->num[0]+grid->num[1]+2;
	for( unsigned i = 0; i < grid->num[2]+1; ++i) {
		grid->vertices[i + new_offset] = grid->vertices[i+grid->offset[2]];
		grid->vertices[i+grid->offset[2]] = -1.0;
	}
	grid->offset[2] = grid->num[0]+1 +grid->num[1]+1;

	grid->numXY = grid->num[0]*grid->num[1];
}

unsigned calcIndex(const GridBins* const grid, const int* const indices ) {
    return indices[0] + indices[1]*getNumBins(grid,0) + indices[2]*getNumBins(grid,0)*getNumBins(grid,1);
}

void calcIJK(const GridBins* const grid, unsigned index, unsigned* indices ) {
    indices[0] = 0; indices[1] = 0; indices[2] = 0;

    unsigned offsets[3];
    offsets[0] = 1;
    offsets[1] = getNumBins(grid,0);
    offsets[2] = getNumBins(grid,0)*getNumBins(grid,1);

    for( int d = 2; d > -1; --d ) {
        unsigned current_offset = offsets[ d ];
        indices[d] = index / current_offset;
        index -= indices[d] * current_offset;
    }
}

int getDimIndex(const GridBins* const grid, unsigned dim, double pos ) {
     // returns -1 for one neg side of mesh
     // and number of bins on the pos side of the mesh
     // need to call isIndexOutside(dim, grid, index) to check if the
     // index is in the mesh

	int dim_index;
	if( pos <= min(grid, dim) ) {
		dim_index = -1;
	} else if( pos >= max(grid, dim)  ) {
		dim_index = getNumBins(grid, dim);
	} else {
		dim_index = ( pos -  min(grid, dim) ) / grid->delta[dim];
	}
	return dim_index;
}

bool isIndexOutside(const GridBins* const grid, unsigned dim, int i) {
	if( i < 0 ||  i >= getNumBins(grid, dim) ) return true;
	return false;
}

bool isOutside(const GridBins* const grid, const int* indices ) {
    for( unsigned d=0; d<3; ++d){
       if( isIndexOutside(grid, d, indices[d]) ) return true;
    }
    return false;
}

unsigned getIndex(const GridBins* const grid, const Position_t& particle_pos) {

    int indices[3]= {0, 0, 0};
    for( unsigned d = 0; d < 3; ++d ) {
        indices[d] = getDimIndex(grid, d, particle_pos[d] );

        // outside the grid
        if( isIndexOutside(grid, d, indices[d] ) ) { return UINT_MAX; }
    }

    return calcIndex(grid, indices );
}

unsigned getMaxNumVertices(const GridBins* const grid) {
	return MAXNUMVERTICES;
}

unsigned getNumCells( const GridBins* const grid ) {
	return grid->num[0]*grid->num[1]*grid->num[2];
}

void getCenterPointByIndices(const GridBins* const grid, unsigned* indices,  Position_t& pos ){
	for( unsigned i=0; i<3; ++i) {
		pos[i] = (getVertex(grid, i, indices[i]) + getVertex(grid, i, indices[i]+1)) / 2.0f ;
	}
}


void getCenterPointByIndex(const GridBins* const grid, unsigned index, Position_t& pos ){
	unsigned indices[3];
	calcIJK(grid, index, indices);

	getCenterPointByIndices( grid, indices, pos);
}

float_t getDistance( Position_t& pos1, Position_t& pos2) {
	float_t deltaSq[3];
	deltaSq[0] = (pos1[0] - pos2[0])*(pos1[0] - pos2[0]);
	deltaSq[1] = (pos1[1] - pos2[1])*(pos1[1] - pos2[1]);
	deltaSq[2] = (pos1[2] - pos2[2])*(pos1[2] - pos2[2]);
	return sqrt( deltaSq[0] + deltaSq[1] + deltaSq[2]);
}

void getDistancesToAllCenters(const GridBins* const grid, Position_t& pos, float_t* distances) {
	unsigned index = 0;
	unsigned indices[3];
	for( unsigned i = 0; i < grid->num[0]; ++i ) {
		indices[0] = i;
		for( unsigned j = 0; j < grid->num[1]; ++j ) {
			indices[1] = j;
			for( unsigned k = 0; k < grid->num[2]; ++k ) {
				indices[2] = k;
				Position_t pixelPoint;
				getCenterPointByIndices(grid, indices, pixelPoint);
				distances[index] = getDistance( pixelPoint, pos );
				++index;
			}
		}
	}
}

GridBinsHost::GridBinsHost( float_t negX, float_t posX, unsigned nX,
	                        float_t negY, float_t posY, unsigned nY,
	                        float_t negZ, float_t posZ, unsigned nZ){
	ptr = (GridBins*) malloc( sizeof(GridBins) );
	ctor(ptr);
	setVertices(ptr, 0, negX, posX, nX);
	setVertices(ptr, 1, negY, posY, nY);
	setVertices(ptr, 2, negZ, posZ, nZ);
	finalize(ptr);
	ptr_device = NULL;
	temp = NULL;
	cudaCopyMade = false;
}

GridBinsHost::~GridBinsHost(){
	free( ptr );
#ifdef CUDA
	if( cudaCopyMade ) {
		if( ptr_device != NULL ) {
			cudaFree( ptr_device );
		}
	}
#endif
}

void GridBinsHost::copyToGPU(void) {
#ifdef CUDA
	cudaCopyMade = true;
	CUDA_CHECK_RETURN( cudaMalloc( &ptr_device, sizeof(GridBins) ));
	CUDA_CHECK_RETURN( cudaMemcpy(ptr_device, ptr, sizeof(GridBins), cudaMemcpyHostToDevice ));
#endif
}


unsigned GridBinsHost::getIndex(float_t x, float_t y, float_t z) const {
	Position_t pos(x,y,z);
	return MonteRay::getIndex( ptr, pos);
}

}

