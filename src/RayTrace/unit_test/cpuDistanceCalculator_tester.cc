#include <UnitTest++.h>

#include "MonteRayConstants.hh"
#include "GridBins.h"
#include "cpuRayTrace.h"

using namespace MonteRay;

SUITE( DistanceCalculatorCPUTest ) {

	class DistanceCalculatorCPUTest {
	public:

		DistanceCalculatorCPUTest(){
			grid = (GridBins*) malloc( sizeof(GridBins) );
			ctor( grid );
			setVertices(grid, 0, 0.0, 10.0, 10);
			setVertices(grid, 1, 0.0, 10.0, 10);
			setVertices(grid, 2, 0.0, 10.0, 10);
			finalize(grid);

			originalPos[0] = 5.0;
			originalPos[1] = 5.0;
			originalPos[2] = 5.0;

			numCells = getNumCells(grid);
			distances = (float_t*) malloc( sizeof(float_t)* numCells );
		}

		~DistanceCalculatorCPUTest(){
			free( grid );
		}

		GridBins* grid;
		Position_t originalPos;
		float_t* distances;
		unsigned numCells;
	};

	TEST_FIXTURE(DistanceCalculatorCPUTest, calculateDistances)
	{
		getDistancesToAllCenters( grid, originalPos, distances );

		float_t distance0 = sqrt( 4.5f*4.5f*3 );
		CHECK_CLOSE( distance0, distances[0], 1e-7 );
	}

}
