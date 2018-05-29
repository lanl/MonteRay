#include "GridBins.hh"

#include <iostream>
#include <fstream>
#include <ostream>

#include "GPUErrorCheck.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRay_binaryIO.hh"
#include "BinarySearch.hh"
#include "MonteRayConstants.hh"

#ifdef MCATK_INLINED
#include "ReadLnk3dnt.hh"
#endif

namespace MonteRay{

CUDA_CALLABLE_MEMBER
void GridBins::setVertices( unsigned dim, float_t min, float_t max, unsigned numBins ) {

	minMax[dim*2] = min;
	minMax[dim*2+1] = max;

	delta[dim] = (max - min) / numBins;
	num[dim] = numBins;

	if( numBins+1 > MAXNUMVERTICES ) {
		ABORT("GridBins::setVertices -- exceeding max number of vertices.");
	}

	vertices[ offset[dim] ] = min;

	//global::float_t location;
	for( unsigned i = 1; i<numBins+1; ++i) {
		vertices[ i+ offset[dim]] = vertices[ i - 1 + offset[dim] ] + delta[dim];
	}
	regular[dim] = true;
}

CUDA_CALLABLE_MEMBER
void GridBins::finalize() {
	for( unsigned dim = 0; dim < 3; ++dim) {
		if( num[dim] == 0 ) {
			ABORT("GridBins::finalize -- vertices not set.");
		}
	}

	// move y data
	unsigned int pad = 1;

	unsigned new_offset = num[0] + pad;
	for( unsigned i = 0; i < num[1]+1; ++i) {
		vertices[i + new_offset] = vertices[i+offset[1]];
		vertices[i+offset[1]] = -1.0;
	}
	offset[1] = num[0] + pad;

	// move z data
	new_offset = num[0] + num[1] + pad + pad;
	for( unsigned i = 0; i < num[2]+1; ++i) {
		vertices[i + new_offset] = vertices[i+offset[2]];
		vertices[i+offset[2]] = -1.0;
	}
	offset[2] = num[0] + num[1] + pad + pad;

	numXY = num[0]*num[1];
}

CUDA_CALLABLE_MEMBER
unsigned
GridBins::calcIndex(const int* const indices ) const {
    return indices[0] + indices[1]*getNumBins(0) + indices[2]*getNumBins(0)*getNumBins(1);
}

void GridBins::calcIJK(unsigned index, unsigned* indices ) const {
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

CUDA_CALLABLE_MEMBER
int
GridBins::getDimIndex(const unsigned dim, const gpuRayFloat_t pos ) const {
     // returns -1 for one neg side of mesh
     // and number of bins on the pos side of the mesh
     // need to call isIndexOutside(dim, grid, index) to check if the
     // index is in the mesh
	int dim_index;
	gpuFloatType_t minimum = min(dim);
	unsigned numBins = getNumBins(dim);

	if( pos <= minimum ) {
		dim_index = -1;
	} else if( pos >= max(dim)  ) {
		dim_index = numBins;
	} else {
		if( regular[dim] ) {
			dim_index = ( pos -  minimum ) / delta[dim];
		} else {
			dim_index = LowerBoundIndex( vertices + offset[dim], numBins+1, pos  );
		}
	}
	return dim_index;
}

CUDA_CALLABLE_MEMBER
bool GridBins::isIndexOutside( unsigned dim, int i) const {
	return ( (i < 0 ||  i >= getNumBins(dim)) ? true : false);
}

CUDA_CALLABLE_MEMBER
bool GridBins::isOutside(const int* indices ) const {
    if( isIndexOutside(0, indices[0]) ) return true;
    if( isIndexOutside(1, indices[1]) ) return true;
    if( isIndexOutside(2, indices[2]) ) return true;
    return false;
}

CUDA_CALLABLE_MEMBER
unsigned
GridBins::getIndex(const Position_t& particle_pos) {

    int indices[3]= {0, 0, 0};
    for( unsigned d = 0; d < 3; ++d ) {
        indices[d] = getDimIndex(d, particle_pos[d] );

        // outside the grid
        if( isIndexOutside( d, indices[d] ) ) { return UINT_MAX; }
    }

    return calcIndex( indices );
}

Position_t GridBins::getCenterPointByIndices( const unsigned* const indices ) const{
	Position_t pos;
	for( unsigned i=0; i<3; ++i) {
		const unsigned vertexIndex = indices[i];
		pos[i] = (getVertex(i, vertexIndex) + getVertex(i, vertexIndex+1)) / 2.0f ;
	}
	return pos;
}

Position_t GridBins::getCenterPointByIndex(unsigned index ) const{
	unsigned indices[3];
	calcIJK(index, indices);

	return getCenterPointByIndices( indices );
}

float_t getDistance( Position_t& pos1, Position_t& pos2) {
	float_t deltaSq[3];
	deltaSq[0] = (pos1[0] - pos2[0])*(pos1[0] - pos2[0]);
	deltaSq[1] = (pos1[1] - pos2[1])*(pos1[1] - pos2[1]);
	deltaSq[2] = (pos1[2] - pos2[2])*(pos1[2] - pos2[2]);
	return sqrt( deltaSq[0] + deltaSq[1] + deltaSq[2]);
}

GridBinsHost::GridBinsHost(){
	ptr = new GridBins;
	ptr_device = NULL;
	temp = NULL;
	cudaCopyMade = false;
}

GridBinsHost::GridBinsHost( float_t negX, float_t posX, unsigned nX,
	                        float_t negY, float_t posY, unsigned nY,
	                        float_t negZ, float_t posZ, unsigned nZ){
	ptr = new GridBins;

	ptr->setVertices( 0, negX, posX, nX);
	ptr->setVertices( 1, negY, posY, nY);
	ptr->setVertices( 2, negZ, posZ, nZ);
	ptr->finalize();

	ptr_device = NULL;
	temp = NULL;
	cudaCopyMade = false;
}

GridBinsHost::GridBinsHost( std::vector<double> x, std::vector<double> y, std::vector<double> z) {
	ptr = new GridBins;

    setVertices(0, x );
    setVertices(1, y );
    setVertices(2, z );
    finalize();

	ptr_device = NULL;
	temp = NULL;
	cudaCopyMade = false;
}

GridBinsHost::~GridBinsHost(){
	delete ptr;
#ifdef __CUDACC__
	if( cudaCopyMade ) {
		if( ptr_device != NULL ) {
			cudaFree( ptr_device );
		}
	}
#endif
}

void GridBinsHost::copyToGPU(void) {
#ifdef __CUDACC__
	cudaCopyMade = true;
	CUDA_CHECK_RETURN( cudaMalloc( &ptr_device, sizeof(GridBins) ));
	CUDA_CHECK_RETURN( cudaMemcpy(ptr_device, ptr, sizeof(GridBins), cudaMemcpyHostToDevice ));
#endif
}


unsigned GridBinsHost::getIndex(float_t x, float_t y, float_t z) const {
	return ptr->getIndex( Position_t(x,y,z) );
}


//struct GridBins {
//	float_t vertices[MAXNUMVERTICES*3];
//
//	unsigned num[3];
//	unsigned numXY;
//
//	unsigned offset[3];
//
//	float_t delta[3];
//
//	float_t minMax[6];
//
//	int isRegular[3];
//};

void GridBinsHost::write(std::ostream& outf) const{
    binaryIO::write(outf, ptr->num[0] );
    binaryIO::write(outf, ptr->num[1] );
    binaryIO::write(outf, ptr->num[2] );
    binaryIO::write(outf, ptr->numXY );
    binaryIO::write(outf, ptr->offset[0] );
    binaryIO::write(outf, ptr->offset[1] );
    binaryIO::write(outf, ptr->offset[2] );
    binaryIO::write(outf, ptr->regular[0] );
    binaryIO::write(outf, ptr->regular[1] );
    binaryIO::write(outf, ptr->regular[2] );
    binaryIO::write(outf, ptr->delta[0] );
    binaryIO::write(outf, ptr->delta[1] );
    binaryIO::write(outf, ptr->delta[2] );
    binaryIO::write(outf, ptr->minMax[0] );
    binaryIO::write(outf, ptr->minMax[1] );
    binaryIO::write(outf, ptr->minMax[2] );
    binaryIO::write(outf, ptr->minMax[3] );
    binaryIO::write(outf, ptr->minMax[4] );
    binaryIO::write(outf, ptr->minMax[5] );

    for( unsigned i=0; i< MAXNUMVERTICES*3; ++i ) {
    	binaryIO::write( outf, ptr->vertices[i] );
    }
}

void GridBinsHost::read(std::istream& infile) {
    binaryIO::read(infile, ptr->num[0] );
    binaryIO::read(infile, ptr->num[1] );
    binaryIO::read(infile, ptr->num[2] );
    binaryIO::read(infile, ptr->numXY );
    binaryIO::read(infile, ptr->offset[0] );
    binaryIO::read(infile, ptr->offset[1] );
    binaryIO::read(infile, ptr->offset[2] );
    binaryIO::read(infile, ptr->regular[0] );
    binaryIO::read(infile, ptr->regular[1] );
    binaryIO::read(infile, ptr->regular[2] );
    binaryIO::read(infile, ptr->delta[0] );
    binaryIO::read(infile, ptr->delta[1] );
    binaryIO::read(infile, ptr->delta[2] );
    binaryIO::read(infile, ptr->minMax[0] );
    binaryIO::read(infile, ptr->minMax[1] );
    binaryIO::read(infile, ptr->minMax[2] );
    binaryIO::read(infile, ptr->minMax[3] );
    binaryIO::read(infile, ptr->minMax[4] );
    binaryIO::read(infile, ptr->minMax[5] );

    for( unsigned i=0; i< MAXNUMVERTICES*3; ++i ) {
    	binaryIO::read( infile, ptr->vertices[i] );
    }
}


void GridBinsHost::write( const std::string& filename ) const {
    std::ofstream outfile;

    outfile.open( filename.c_str(), std::ios::binary | std::ios::out);
    if( ! outfile.is_open() ) {
        fprintf(stderr, "GridBinsHost::write -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( outfile.good() );
    outfile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    write( outfile );
    outfile.close();
}

void GridBinsHost::read( const std::string& filename ) {
    std::ifstream infile;
    if( infile.is_open() ) {
        infile.close();
    }
    infile.open( filename.c_str(), std::ios::binary | std::ios::in);

    if( ! infile.is_open() ) {
        fprintf(stderr, "GridBinsHost::read -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( infile.good() );
    infile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    read(infile);
    infile.close();
}

#ifdef MCATK_INLINED
void GridBinsHost::loadFromLnk3dnt( const std::string& filename ){
    ReadLnk3dnt file( filename);
    if( file.getGeometryString() != "XYZ" )  {
    	throw std::runtime_error( "Invalid Lnk3dnt type -- MonteRay can only support XYZ");
    }

    for( unsigned d=0; d < 3; ++d) {
    	std::vector<double> vertices = file.getVertices(d);
    	setVertices(d, vertices );
    }
    finalize();
}
#endif

CUDA_CALLABLE_MEMBER
unsigned GridBins::rayTrace( int* global_indices, gpuRayFloat_t* distances, const Position_t& pos, const Position_t& dir, float_t distance,  bool outsideDistances) const {
	const bool debug = false;

    int current_indices[3] = {0, 0, 0}; // current position indices in the grid, must be int because can be outside

    if( debug ){
    	printf( "GridBins::rayTrace --------------------------------\n");
    }

    int cells[3][MAXNUMVERTICES];
    gpuRayFloat_t crossingDistances[3][MAXNUMVERTICES];
    unsigned numCrossings[3];

    for( unsigned i=0; i<3; ++i){
    	current_indices[i] = getDimIndex(i, pos[i] );

    	numCrossings[i] = calcCrossings( vertices + offset[i], num[i]+1, cells[i], crossingDistances[i], pos[i], dir[i], distance, current_indices[i]);

    	if( debug ){
    		printf( "GridBins::rayTrace -- current_indices[i]=%d\n", current_indices[i] );
    		printf( "GridBins::rayTrace -- numCrossings[i]=%d\n", numCrossings[i] );
    	}

        // if outside and ray doesn't move inside then ray never enters the grid
        if( isIndexOutside(i,current_indices[i]) && numCrossings[i] == 0  ) {
            return 0U;
        }
    }

    if( debug ){
    	printf( "GridBins::rayTrace -- numCrossings[0]=%d\n", numCrossings[0] );
    	printf( "GridBins::rayTrace -- numCrossings[1]=%d\n", numCrossings[1] );
    	printf( "GridBins::rayTrace -- numCrossings[2]=%d\n", numCrossings[2] );
    }

    return orderCrossings(global_indices, distances, MAXNUMVERTICES, cells[0], crossingDistances[0], numCrossings, current_indices, distance, outsideDistances);
}


CUDA_CALLABLE_MEMBER
unsigned calcCrossings(const float_t* const vertices, unsigned nVertices, int* cells, gpuRayFloat_t* distances, float_t pos, float_t dir, float_t distance, int index ){
	const bool debug = false;

	unsigned nDistances = 0;

	if( debug ) {
		printf( "GridBins::calcCrossings --------------------------------\n" );
		printf( "calcCrossings -- vertices[0]=%f\n", vertices[0] );
		printf( "calcCrossings -- vertices[nVertices-1]=%f\n", vertices[nVertices-1] );
		printf( "calcCrossings -- pos=%f\n", pos );
		printf( "calcCrossings -- dir=%f\n", dir );
	}

    if( abs(dir) <= MonteRay::epsilon ) {
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
        	printf( " calcCrossings -- current_index=%d\n", current_index );
        	printf( " calcCrossings --        offset=%d\n", offset );
        	printf( " calcCrossings -- vertices[current_index + offset]=%f\n", vertices[current_index + offset] );
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
    		printf( " calcCrossings -- i=%d  cell index=%d  distance=%f\n", i, cells[i], distances[i] );
    	}
    	printf( "-----------------------------------------------------------------------\n" );
    }

    return nDistances;
}

CUDA_CALLABLE_MEMBER
unsigned GridBins::orderCrossings(int* global_indices, gpuRayFloat_t* distances, unsigned num, const int* const cells, const gpuRayFloat_t* const crossingDistances, unsigned* numCrossings, int* indices, float_t distance, bool outsideDistances ) const {
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
    		printf( "Debug: i=%d  numCrossings=%d\n", i, numCrossings[i]);
    		for( unsigned j=0; j< numCrossings[i]; ++j ) {
    			printf( "Debug: j=%d  index=%d  distance=%f", j, *((cells+i*num) + j),  *((crossingDistances+i*num)+j) );
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
        	printf( "Debug: minDim=%d  index=%d   minimumDistance=%f\n", minDim, indices[minDim], minimumDistance);
        }

        // test for outside of the grid
        outside = isOutside( indices );

        if( debug ) {
            if( outside ) {
            	printf( "Debug: ray is outside \n" );
            } else {
            	printf( "Debug: ray is inside \n" );
            }
        }

        float_t currentDistance = minimumDistance;

        if( !outside || outsideDistances ) {
        	float_t deltaDistance = currentDistance - priorDistance;

            if( deltaDistance > 0.0  ) {
                unsigned global_index;
                if( !outside ) {
                    global_index = calcIndex(indices );
                } else {
                    global_index = UINT_MAX;
                }
                global_indices[numRayCrossings] = global_index;
                distances[numRayCrossings] = deltaDistance;
                ++numRayCrossings;

                if( debug ) {
                	printf( "Debug: ******************\n" );
                	printf( "Debug:  Entry Num    = %d\n", numRayCrossings );
                	printf( "Debug:     index[0]  = %d\n", indices[0] );
                	printf( "Debug:     index[1]  = %d\n", indices[1] );
                	printf( "Debug:     index[2]  = %d\n", indices[2] );
                	printf( "Debug:     distance  = %f\n", deltaDistance );
                }
            }
        }

        if( currentDistance >= distance ) {
            break;
        }

        indices[minDim] = *((cells+minDim*num) + start[minDim]+1);

        if( ! outside ) {
            if( isIndexOutside(minDim, indices[minDim] ) ) {
                // ray has moved outside of grid
                break;
            }
        }

        ++start[minDim];
        priorDistance = currentDistance;
    }

    return numRayCrossings;
}

CUDA_CALLABLE_KERNEL
void
kernelRayTrace(void* ptrNumCrossings,
		GridBins* ptrGrid,
		int* ptrCells,
		gpuRayFloat_t* ptrDistances,
		gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z,
		gpuFloatType_t u, gpuFloatType_t v, gpuFloatType_t w,
		gpuFloatType_t distance,
		bool outsideDistances) {

	const bool debug = false;

	if( debug ) {
		printf("kernelRayTrace(GridBins*):: Starting kernelRayTrace ******************\n");
	}

	unsigned* numCrossings = (unsigned*) ptrNumCrossings;

	Position_t pos( x, y, z );
	Direction_t dir( u, v, w );

	numCrossings[0] = ptrGrid->rayTrace( ptrCells, ptrDistances, pos, dir, distance, outsideDistances);

	if( debug ) {
		printf("kernelRayTrace(GridBins*):: numCrossings=%d\n",numCrossings[0]);
	}
}

}

