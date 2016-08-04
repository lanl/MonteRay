#ifndef CPUCROSSINGDISTANCE_H_
#define CPUCROSSINGDISTANCE_H_
#include "global.h"

namespace MonteRay{

class CartesianGrid;
class CrossingDistance {
public:
	typedef global::float_t float_t;

	CrossingDistance() {}

	~CrossingDistance() {}


	static unsigned calc(const float_t* const vertices, unsigned nVertices, int* cells, float_t* distances, float_t pos, float_t dir, float_t distance, int index );
	static unsigned orderCrossings(const CartesianGrid* const grid, int* global_indices, float_t* distances, unsigned num, const int* const cells, const float_t* const crossingDistances, unsigned* numCrossings, int* indices, double distance, bool outsideDistances );

private:

};

}

#endif /* CPUCROSSINGDISTANCE_H_ */
