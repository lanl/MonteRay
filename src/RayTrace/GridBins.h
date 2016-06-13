#ifndef GRIDBINS_H_
#define GRIDBINS_H_

#include <limits.h>
#include <stdio.h>        /* perror */
#include <errno.h>        /* errno */
#include <stdlib.h>

#define MAXNUMVERTICES 1001

#include "global.h"
#include "Vector3D.h"

typedef global::float_t float_t;
typedef Vector3D Position_t;
typedef Vector3D Direction_t;

struct GridBins {
	float_t vertices[MAXNUMVERTICES*3];

	unsigned num[3];
	unsigned numXY;

	unsigned offset[3];

	float_t delta[3];

	float_t minMax[6];
};

void ctor(GridBins* grid);

unsigned getNumVertices(const GridBins* const grid, unsigned dim);

unsigned getNumXY(const GridBins* const grid);

unsigned getNumBins(const GridBins* const grid, unsigned dim);

unsigned getNumBins(const GridBins* const grid, unsigned dim, unsigned index);

float_t getVertex(const GridBins* const grid, unsigned dim, unsigned index );

float_t min(const GridBins* const grid, unsigned dim);

float_t max(const GridBins* const grid, unsigned dim);

void setVertices( GridBins* grid, unsigned dim, global::float_t min, float max, unsigned numBins );

void finalize(GridBins* grid);

unsigned calcIndex(const GridBins* const grid, const int* const indices );

void calcIJK(const GridBins* const grid, unsigned index, unsigned* indices );

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

#endif /* GRIDBINS_H_ */
