#ifndef GRIDBINS_H_
#define GRIDBINS_H_

#include <limits.h>
#include <stdio.h>        /* perror */
#include <errno.h>        /* errno */
#include <stdlib.h>
#include <vector>

#define MAXNUMVERTICES 1001

#include "MonteRayDefinitions.hh"
#include "MonteRayVector3D.hh"

namespace MonteRay{

typedef gpuFloatType_t float_t;
typedef MonteRay::Vector3D<double> Position_t;
typedef MonteRay::Vector3D<double> Direction_t;

struct GridBins {
	float_t vertices[MAXNUMVERTICES*3];

	unsigned num[3];
	unsigned numXY;

	unsigned offset[3];

	float_t delta[3];

	float_t minMax[6];

	int isRegular[3];
};

void ctor(GridBins* grid);

unsigned getNumVertices(const GridBins* const grid, unsigned dim);

unsigned getNumXY(const GridBins* const grid);

unsigned getNumBins(const GridBins* const grid, unsigned dim);

unsigned getNumBins(const GridBins* const grid, unsigned dim, unsigned index);

float_t getVertex(const GridBins* const grid, unsigned dim, unsigned index );

bool isRegular(const GridBins* const grid, unsigned dim);

float_t min(const GridBins* const grid, unsigned dim);

float_t max(const GridBins* const grid, unsigned dim);

void setVertices( GridBins* grid, unsigned dim, std::vector<double> );
void setVertices( GridBins* grid, unsigned dim, float_t min, float_t max, unsigned numBins );

void finalize(GridBins* grid);

unsigned calcIndex(const GridBins* const grid, const int* const indices );

void calcIJK(const GridBins* const grid, unsigned index, unsigned* indices );

unsigned getIndexBinaryFloat( const float_t* const values, unsigned count, float_t value );
int getDimIndex(const GridBins* const grid, unsigned dim, double pos );

bool isIndexOutside(const GridBins* const grid, unsigned dim, int i);

bool isOutside(const GridBins* const grid, const int* indices );

unsigned getIndex(const GridBins* const grid, const Position_t& particle_pos);

unsigned getMaxNumVertices(const GridBins* const grid);

unsigned getNumCells( const GridBins* const grid );

void getCenterPointByIndices(const GridBins* const grid, unsigned* indices,  Position_t& pos );

void getCenterPointByIndex(const GridBins* const grid, unsigned index, Position_t& pos );

float_t getDistance( Position_t& pos1, Position_t& pos2);

void getDistancesToAllCenters(const GridBins* const grid, Position_t& pos, float_t* distances);


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
		ptr = (GridBins*) malloc( sizeof(GridBins) );
		ctor(ptr);

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

    void setVertices(unsigned dim, std::vector<double> vertices );
    void setVertices(unsigned dim, float_t min, float_t max, unsigned numBins ){
    	MonteRay::setVertices( ptr, dim, min, max, numBins );
    }
    void finalize() {
    	MonteRay::finalize(ptr);
    }
    const GridBins* getPtr() const { return ptr; }
    const GridBins* getPtrDevice() const { return ptr_device; }

    void write(std::ostream& outfile) const;
    void  read(std::istream& infile);

    void write( const std::string& filename ) const;
    void read( const std::string& filename );

    unsigned getNumCells(void) const { return MonteRay::getNumCells(ptr); }
    unsigned getIndex(float_t x, float_t y, float_t z) const;

#ifndef CUDA
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

}
#endif /* GRIDBINS_H_ */
