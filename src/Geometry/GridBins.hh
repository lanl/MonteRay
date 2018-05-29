#ifndef GRIDBINS_H_
#define GRIDBINS_H_

#include <limits.h>
#include <stdio.h>        /* perror */
#include <errno.h>        /* errno */
#include <stdlib.h>
#include <vector>
#include <limits>

#define MAXNUMVERTICES 1001

#include "MonteRayDefinitions.hh"
#include "MonteRayVector3D.hh"
#include "GPUErrorCheck.hh"

namespace MonteRay{

typedef gpuFloatType_t float_t;
typedef MonteRay::Vector3D<gpuRayFloat_t> Position_t;
typedef MonteRay::Vector3D<gpuRayFloat_t> Direction_t;

class GridBins {
public:
	float_t vertices[MAXNUMVERTICES*3];
	unsigned num[3];
	unsigned numXY;

	unsigned offset[3];

	float_t delta[3];

	float_t minMax[6];

	int regular[3];

public:

	CUDA_CALLABLE_MEMBER
	GridBins(){
		initialize();
	}

	CUDA_CALLABLE_MEMBER
	void initialize() {
		offset[0] = 0;
		offset[1] = MAXNUMVERTICES;
		offset[2] = MAXNUMVERTICES*2;
		num[0] = 0;
		num[1] = 0;
		num[2] = 0;
		numXY = 0;
		regular[0] = true;
		regular[1] = true;
		regular[2] = true;
	}

	CUDA_CALLABLE_MEMBER
	unsigned getMaxNumVertices() const {
		return MAXNUMVERTICES;
	}

	CUDA_CALLABLE_MEMBER
	unsigned getOffset( const unsigned dim ) const {
		MONTERAY_ASSERT( dim < 3);
		return offset[dim];
	}

	CUDA_CALLABLE_MEMBER
	void setVertices(unsigned dim, float_t min, float_t max, unsigned numBins);

	template<typename T>
	void setVertices(const unsigned dim, const std::vector<T>& vertices );

	CUDA_CALLABLE_MEMBER
	float_t getVertex(const unsigned dim, const unsigned index ) const {
		return vertices[ offset[dim] + index ];
	}

	CUDA_CALLABLE_MEMBER
	unsigned getNumVertices(const unsigned dim) const {
		MONTERAY_ASSERT( dim < 3);
		return num[dim]+1;
	}

	CUDA_CALLABLE_MEMBER
	unsigned getNumBins(unsigned dim) const {
		MONTERAY_ASSERT( dim < 3);
		return num[dim];
	}

	CUDA_CALLABLE_MEMBER
	bool isRegular( unsigned dim) const {
		MONTERAY_ASSERT( dim < 3);
		return regular[dim];
	}

	CUDA_CALLABLE_MEMBER
	void finalize();

	CUDA_CALLABLE_MEMBER
	unsigned getNumXY() const { return numXY; }

	CUDA_CALLABLE_MEMBER
	float_t min(const unsigned dim) const {
		MONTERAY_ASSERT( dim < 3);
		return minMax[dim*2];
	}

	CUDA_CALLABLE_MEMBER
	float_t max(const unsigned dim) const {
		MONTERAY_ASSERT( dim < 3);
		return minMax[dim*2+1];
	}

	CUDA_CALLABLE_MEMBER
	int getDimIndex(const unsigned dim, const gpuRayFloat_t pos ) const;

	CUDA_CALLABLE_MEMBER
	unsigned getIndex(const Position_t& particle_pos);

	CUDA_CALLABLE_MEMBER
	bool isIndexOutside(unsigned dim, int i) const;

	CUDA_CALLABLE_MEMBER
	unsigned calcIndex(const int* const indices ) const;

	CUDA_CALLABLE_MEMBER
	bool isOutside(const int* indices ) const;

	unsigned getNumCells() const { return num[0]*num[1]*num[2]; }

	Position_t getCenterPointByIndex(unsigned index ) const;
	Position_t getCenterPointByIndices( const unsigned* const indices ) const;

	void calcIJK(unsigned index, unsigned* indices ) const;

	CUDA_CALLABLE_MEMBER
	unsigned rayTrace(int* global_indices, gpuRayFloat_t* distances, const Position_t& pos, const Position_t& dir, float_t distance,  bool outsideDistances) const;

	CUDA_CALLABLE_MEMBER
	unsigned orderCrossings(int* global_indices, gpuRayFloat_t* distances, unsigned num, const int* const cells, const gpuRayFloat_t* const crossingDistances, unsigned* numCrossings, int* indices, float_t distance, bool outsideDistances ) const;

};

//  static methods
float_t getDistance( Position_t& pos1, Position_t& pos2);

CUDA_CALLABLE_MEMBER
unsigned calcCrossings(const float_t* const vertices, unsigned nVertices, int* cells, gpuRayFloat_t* distances, float_t pos, float_t dir, float_t distance, int index );

CUDA_CALLABLE_KERNEL
void
kernelRayTrace(
		void* ptrNumCrossings,
		GridBins* ptrGrid,
		int* ptrCells,
		gpuRayFloat_t* ptrDistances,
		gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z,
		gpuFloatType_t u, gpuFloatType_t v, gpuFloatType_t w,
		gpuFloatType_t distance,
		bool outsideDistances);

class GridBinsHost {
public:
	GridBinsHost();
	GridBinsHost( float_t negX, float_t posX, unsigned nX,
			      float_t negY, float_t posY, unsigned nY,
			      float_t negZ, float_t posZ, unsigned nZ);
	GridBinsHost( std::vector<double> x, std::vector<double> y, std::vector<double> z);

	// ctor that takes a class that provides getVertices(unsigned dim)
	template<class T>
	GridBinsHost( T& reader) {
		ptr = new GridBins;

		for( unsigned d=0; d < 3; ++d) {
			std::vector<double> vertices = reader.getVertices(d);
			setVertices(d, vertices );
		}
		finalize();

		ptr_device = NULL;
		temp = NULL;
		cudaCopyMade = false;
	}

    ~GridBinsHost();

    template<typename T>
    void setVertices(unsigned dim, std::vector<T> vertices );

    void setVertices(unsigned dim, float_t min, float_t max, unsigned numBins ){
    	ptr->setVertices( dim, min, max, numBins );
    }
    void finalize() {
    	ptr->finalize();
    }
    const GridBins* getPtr() const { return ptr; }
    const GridBins* getPtrDevice() const { return ptr_device; }

    void write(std::ostream& outfile) const;
    void  read(std::istream& infile);

    void write( const std::string& filename ) const;
    void read( const std::string& filename );

    unsigned getNumCells(void) const { return ptr->getNumCells(); }
    unsigned getIndex(float_t x, float_t y, float_t z) const;

    bool isRegular(unsigned dim) { return ptr->isRegular(dim); }

	float_t min(const unsigned dim) const { return ptr->min(dim); }
	float_t max(const unsigned dim) const { return ptr->max(dim); }

#ifndef __CUDACC__
    void loadFromLnk3dnt( const std::string& filename );
#endif

    void copyToGPU(void);

private:
    GridBins* ptr;
    GridBins* temp;
    bool cudaCopyMade;

public:
    GridBins* ptr_device;

};

template<typename T>
void GridBinsHost::setVertices(unsigned dim, std::vector<T> vertices ){

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
  		ptr->setVertices( dim, vertices.front(), vertices.back(), vertices.size()-1 );
  	} else {
  		ptr->setVertices( dim, vertices );
  	}
}

template<typename T>
void
GridBins::setVertices( const unsigned dim, const std::vector<T>& verts) {
	minMax[dim*2] = verts.front();
	minMax[dim*2+1] = verts.back();

	delta[dim] = -1.0;
	num[dim] = verts.size()-1;

	if( getNumBins(dim) > MAXNUMVERTICES ) {
		ABORT("GridBins::setVertices -- exceeding max number of vertices.");
	}

	unsigned counter = 0;
	for( auto itr = verts.cbegin(); itr != verts.cend(); ++itr) {
		vertices[offset[dim]+counter] = *itr;
		++counter;
	}

	regular[dim] = false;
}



}
#endif /* GRIDBINS_H_ */
