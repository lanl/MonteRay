#include "GridBins.h"

#include <iostream>
#include <fstream>
#include <ostream>
#include <limits>

#include "GPUErrorCheck.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRay_binaryIO.hh"

#ifndef CUDA
#include "ReadLnk3dnt.hh"
#endif

namespace MonteRay{

void ctor(GridBins* grid){
	grid->offset[0] = 0;
	grid->offset[1] = MAXNUMVERTICES;
	grid->offset[2] = MAXNUMVERTICES*2;
	grid->num[0] = 0;
	grid->num[1] = 0;
	grid->num[2] = 0;
	grid->numXY = 0;
	grid->isRegular[0] = true;
	grid->isRegular[1] = true;
	grid->isRegular[2] = true;
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

bool isRegular(const GridBins* const grid, unsigned dim) {
	if( grid->isRegular[dim] ) {
		return true;
	}
	return false;
}

float_t min(const GridBins* const grid, unsigned dim) {
	return grid->minMax[dim*2];
}

float_t max(const GridBins* const grid, unsigned dim) {
	return grid->minMax[dim*2+1];
}

void setVertices( GridBins* grid, unsigned dim, float_t min, float_t max, unsigned numBins ) {

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
	grid->isRegular[dim] = true;
}

void setVertices( GridBins* grid, unsigned dim, std::vector<double> vertices) {
	grid->minMax[dim*2] = vertices.front();
	grid->minMax[dim*2+1] = vertices.back();

	grid->delta[dim] = -1.0;
	unsigned numBins = vertices.size()-1;
	grid->num[dim] = numBins;

	if( numBins+1 > MAXNUMVERTICES ) {
		perror("GridBins::setVertices -- exceeding max number of vertices.");
		exit(1);
	}

	for( unsigned i = 0; i<numBins+1; ++i) {
		double vertex = vertices.at(i);
		grid->vertices[ i+ grid->offset[dim]] = vertex;
	}

	grid->isRegular[dim] = false;
}

void finalize(GridBins* grid) {
	for( unsigned dim = 0; dim < 3; ++dim) {
		if( grid->num[dim] == 0 ) {
			perror("GridBins::finalize -- vertices not set.");
		    exit(1);
		}
	}

	// move y data
	unsigned int pad = 1;

	unsigned new_offset = grid->num[0] + pad;
	for( unsigned i = 0; i < grid->num[1]+1; ++i) {
		grid->vertices[i + new_offset] = grid->vertices[i+grid->offset[1]];
		grid->vertices[i+grid->offset[1]] = -1.0;
	}
	grid->offset[1] = grid->num[0] + pad;

	// move z data
	new_offset = grid->num[0] + grid->num[1] + pad + pad;
	for( unsigned i = 0; i < grid->num[2]+1; ++i) {
		grid->vertices[i + new_offset] = grid->vertices[i+grid->offset[2]];
		grid->vertices[i+grid->offset[2]] = -1.0;
	}
	grid->offset[2] = grid->num[0] + grid->num[1] + pad + pad;

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


unsigned getIndexBinaryFloat(const float_t* const values, unsigned count, float_t value ) {
    // modified from http://en.cppreference.com/w/cpp/algorithm/upper_bound
    unsigned it, step;
    unsigned first = 0U;

    while (count > 0U) {
        it = first;
        step = count / 2;
        it += step;
        if(!(value < values[it])) {
            first = ++it;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    if( first > 0 ) { --first; }
    return first;
}

int getDimIndex(const GridBins* const grid, unsigned dim, double pos ) {
     // returns -1 for one neg side of mesh
     // and number of bins on the pos side of the mesh
     // need to call isIndexOutside(dim, grid, index) to check if the
     // index is in the mesh

	int dim_index;
	float_t minimum = min(grid, dim);
	unsigned numBins = grid->num[dim];

	if( pos <= minimum ) {
		dim_index = -1;
	} else if( pos >= max(grid, dim)  ) {
		dim_index = numBins;
	} else {
		if( grid->isRegular[dim] ) {
			dim_index = ( pos -  minimum ) / grid->delta[dim];
		} else {
			dim_index = getIndexBinaryFloat( grid->vertices + grid->offset[dim], numBins+1, pos  );
		}
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

GridBinsHost::GridBinsHost(){
	ptr = (GridBins*) malloc( sizeof(GridBins) );
	ctor(ptr);
	ptr_device = NULL;
	temp = NULL;
	cudaCopyMade = false;
}

GridBinsHost::GridBinsHost( float_t negX, float_t posX, unsigned nX,
	                        float_t negY, float_t posY, unsigned nY,
	                        float_t negZ, float_t posZ, unsigned nZ){
	ptr = (GridBins*) malloc( sizeof(GridBins) );
	ctor(ptr);

	MonteRay::setVertices(ptr, 0, negX, posX, nX);
	MonteRay::setVertices(ptr, 1, negY, posY, nY);
	MonteRay::setVertices(ptr, 2, negZ, posZ, nZ);
	MonteRay::finalize(ptr);

	ptr_device = NULL;
	temp = NULL;
	cudaCopyMade = false;
}

GridBinsHost::GridBinsHost( std::vector<double> x, std::vector<double> y, std::vector<double> z) {
	ptr = (GridBins*) malloc( sizeof(GridBins) );
	ctor(ptr);

    setVertices(0, x );
    setVertices(1, y );
    setVertices(2, z );
    finalize();

	ptr_device = NULL;
	temp = NULL;
	cudaCopyMade = false;
}

GridBinsHost::~GridBinsHost(){
	free( ptr );
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
#else
	throw std::runtime_error("GridBinsHost::copyToGPU -- Can not copy to GPU without CUDA.")
#endif
}


unsigned GridBinsHost::getIndex(float_t x, float_t y, float_t z) const {
	Position_t pos(x,y,z);
	return MonteRay::getIndex( ptr, pos);
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
    binaryIO::write(outf, ptr->isRegular[0] );
    binaryIO::write(outf, ptr->isRegular[1] );
    binaryIO::write(outf, ptr->isRegular[2] );
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
    binaryIO::read(infile, ptr->isRegular[0] );
    binaryIO::read(infile, ptr->isRegular[1] );
    binaryIO::read(infile, ptr->isRegular[2] );
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

#ifndef CUDA
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

void GridBinsHost::setVertices(unsigned dim, std::vector<double> vertices ){

  	double delta = 0.0;
  	double lastDelta = 99.0;
  	bool uniform = true;
  	for( unsigned i = 1; i< vertices.size(); ++i){
  		delta = vertices.at(i) - vertices.at(i-1);
  		if( i > 1 ) {
//  			std::cout << "Debug:: i = " << i << " delta = " << delta << " lastdelta = " << lastDelta << "\n";
  			double epsilon = 10.0*(std::nextafter( lastDelta,  std::numeric_limits<double>::infinity() ) - lastDelta);
  			if( std::abs(delta-lastDelta) > epsilon ) {
//   				std::cout << "Debug:: delta - lastDelta > epsilon -- diff = " << std::abs(delta-lastDelta) << " epsilon = " << epsilon << "\n";
  				uniform = false;
  				break;
  			}

  		}
  		lastDelta = delta;
  	}

  	if( uniform ) {
  		MonteRay::setVertices( ptr, dim, vertices.front(), vertices.back(), vertices.size()-1 );
  	} else {
  		MonteRay::setVertices( ptr, dim, vertices );
  	}
  }

}

