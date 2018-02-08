/*
 * MonteRay_SpatialGrid.hh
 *
 *  Created on: Feb 5, 2018
 *      Author: jsweezy
 */

#ifndef MONTERAY_SPATIALGRID_HH_
#define MONTERAY_SPATIALGRID_HH_

namespace MonteRay {

class MonteRay_SpatialGrid {
public:
	MonteRay_SpatialGrid();
	virtual ~MonteRay_SpatialGrid(){};

public:
	//TRA/JES Move to GridBins
	static const unsigned OUTSIDE_MESH;
};

} /* namespace MonteRay */

#endif /* MONTERAY_SPATIALGRID_HH_ */
