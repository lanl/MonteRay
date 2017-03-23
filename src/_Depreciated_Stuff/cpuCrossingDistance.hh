#ifndef CPUCROSSINGDISTANCE_H_
#define CPUCROSSINGDISTANCE_H_
#include "MonteRayConstants.hh"

namespace MonteRay{

/**
 * \class CrossingDistance
 *
 * \brief Provides ray tracing functions for calling from the CPU
 *
 * This class is used for testing for proper ray tracing against the
 * GPU ray tracing.  Not really a class either.   Contains static
 * functions.  Not needed or used for applications, only used for
 * the understanding of the developer.
 *
 * \author Jeremy Sweezy
 *
 * Contact: jsweezy@lanl.gov
 */

class CartesianGrid;
class CrossingDistance {
public:

	CrossingDistance() {}

	~CrossingDistance() {}

	/// Calculates the crossing distances to vertices in one dimension.
	/// @param [in]  vertices   An array of the location of vertices in a single dimension
	/// @param [in]  nVertices  Number of vertices
	/// @param [out] cells      An array of cell 1-D indices that the ray crosses
	/// @param [out] distances  An array of 1-D distances crossed by the ray, order matches cells
	/// @param [in]  pos        1-D component of the ray position.
	/// @param [in]  dir        1-D component of the ray direction, between 0.0 and 1.0.
	/// @param [in]  distance   The maximum distance the ray travels
	/// @param [in]  index      Starting index of the ray (A short cut which must be correct, should implement a fix-up when it is wrong).
	static unsigned calc(const float_t* const vertices, unsigned nVertices, int* cells, float_t* distances, float_t pos, float_t dir, float_t distance, int index );


	/// Orders the crossing distances in 3-D.
	/// @param [in]  grid                        A 3-d Cartesian grid of vertices
	/// @param [out] global_indices              An array of cell 3-D indices that the ray crosses
	/// @param [out] distances                   An array of 3-D distances crossed by the ray, order matches global_indices
	/// @param [in]  num                         Max. number of possible crossing distances in each direction.
	/// @param [in]  cells[3][num]               2-D array cell indices of crossings, first dim size=3 for X,Y,Z
	/// @param [in]  crossingDistances[3][num]   2-D array of crossing distances, first dim size=3 for X,Y,Z
	/// @param [in]  numCrossings[3]             1-D array of the number of crossings in each dimension
	/// @param [in]  outsideDistances            Boolean flag the controls the calculation of distances in the space surrounding the mesh
	static unsigned orderCrossings(const CartesianGrid* const grid, int* global_indices, float_t* distances, unsigned num, const int* const cells, const float_t* const crossingDistances, unsigned* numCrossings, int* indices, double distance, bool outsideDistances );

private:

};

}

#endif /* CPUCROSSINGDISTANCE_H_ */
