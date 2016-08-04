#ifndef CPURAYTRACE_H_
#define CPURAYTRACE_H_

#include "global.h"
#include "Vector3D.h"

#include "GridBins.h"

namespace MonteRay{

unsigned calcCrossings(const float_t* const vertices, unsigned nVertices, int* cells, float_t* distances, float_t pos, float_t dir, float_t distance, int index );
unsigned orderCrossings(const GridBins* const grid, int* global_indices, float_t* distances, unsigned num, const int* const cells, const float_t* const crossingDistances, unsigned* numCrossings, int* indices, float_t distance, bool outsideDistances );
unsigned rayTrace(const GridBins* const grid, int* global_indices, float_t* distances, const Position_t& pos, const Position_t& dir, float_t distance,  bool outsideDistances);

}

#endif /* CPURAYTRACE_H_ */
