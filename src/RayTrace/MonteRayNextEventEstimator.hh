#ifndef MONTERAYNEXTEVENTESTIMATOR_HH_
#define MONTERAYNEXTEVENTESTIMATOR_HH_

#include "MonteRayDefinitions.hh"
#include "MonteRayCopyMemory.hh"
#include "GPUErrorCheck.hh"

#include "Ray.hh"
#include "gpuRayTrace.h"
#include "GridBins.h"
#include "MonteRay_MaterialProperties.hh"
#include "MonteRayMaterialList.hh"
#include "ExpectedPathLength.h"

namespace MonteRay {

class MonteRayNextEventEstimator : public CopyMemoryBase<MonteRayNextEventEstimator> {
public:
	typedef gpuTallyType_t tally_t;
	typedef gpuFloatType_t position_t;
	using Base = MonteRay::CopyMemoryBase<MonteRayNextEventEstimator> ;

	CUDAHOST_CALLABLE_MEMBER MonteRayNextEventEstimator(unsigned num){
		if( num == 0 ) { num = 1; }
		if( Base::debug ) {
			std::cout << "RayList_t::RayList_t(n), n=" << num << " \n";
		}
		init();
		x = (position_t*) MonteRayHostAlloc( num*sizeof( position_t ), Base::isManagedMemory );
		y = (position_t*) MonteRayHostAlloc( num*sizeof( position_t ), Base::isManagedMemory );
		z = (position_t*) MonteRayHostAlloc( num*sizeof( position_t ), Base::isManagedMemory );
		tally = (tally_t*) MonteRayHostAlloc( num*sizeof( tally_t ), Base::isManagedMemory );
		nAllocated = num;
	}

	CUDAHOST_CALLABLE_MEMBER ~MonteRayNextEventEstimator(){}

	CUDAHOST_CALLABLE_MEMBER void init() {
		nUsed = 0;
		nAllocated = 0;
		radius = 0.0;
		x = NULL;
		y = NULL;
		tally = NULL;

		pGridBinsHost = NULL;
		pGridBins = NULL;
		pMatPropsHost = NULL;
		pMatProps = NULL;
		pMatListHost = NULL;
		pMatList = NULL;
	}

	CUDAHOST_CALLABLE_MEMBER void copy(const MonteRayNextEventEstimator* rhs) {
		if( Base::debug ) {
			std::cout << "Debug: MonteRayNextEventEstimator::copy (const MonteRayNextEventEstimator* rhs) \n";
		}

		if( Base::isCudaIntermediate && rhs->isCudaIntermediate ) {
			throw std::runtime_error(" MonteRayNextEventEstimator::copy -- can NOT copy CUDA intermediate to CUDA intermediate.");
		}

		if( !Base::isCudaIntermediate && !rhs->isCudaIntermediate ) {
			throw std::runtime_error(" MonteRayNextEventEstimator::copy -- can NOT copy CUDA non-intermediate to CUDA non-intermediate.");
		}

		if( nAllocated > 0 && nAllocated != rhs->nAllocated) {
			throw std::runtime_error(" MonteRayNextEventEstimator::copy -- can NOT change the number of allocated tally points.");
		}

		unsigned num = rhs->nAllocated;
		if( Base::isCudaIntermediate ) {
			// target is the intermediate, origin is the host
			if( x == NULL ) {
				x = (position_t*) MonteRayDeviceAlloc( num*sizeof( position_t ) );
				y = (position_t*) MonteRayDeviceAlloc( num*sizeof( position_t ) );
				z = (position_t*) MonteRayDeviceAlloc( num*sizeof( position_t ) );
				tally = (tally_t*) MonteRayDeviceAlloc( num*sizeof( tally_t ) );
			}
			MonteRayMemcpy(x, rhs->x, num*sizeof(position_t), cudaMemcpyHostToDevice);
			MonteRayMemcpy(y, rhs->y, num*sizeof(position_t), cudaMemcpyHostToDevice);
			MonteRayMemcpy(z, rhs->z, num*sizeof(position_t), cudaMemcpyHostToDevice);
			MonteRayMemcpy(tally, rhs->tally, num*sizeof(tally_t), cudaMemcpyHostToDevice);
			pGridBins = pGridBinsHost->ptr_device;
			pMatProps = pMatPropsHost->ptrData_device;
			pMatList = pMatListHost->ptr_device;
			pHash = pHashHost->getPtrDevice();
		} else {
			// target is the host, origin is the intermediate
			MonteRayMemcpy(rhs->x, x, num*sizeof(position_t), cudaMemcpyDeviceToHost);
			MonteRayMemcpy(rhs->y, y, num*sizeof(position_t), cudaMemcpyDeviceToHost);
			MonteRayMemcpy(rhs->z, z, num*sizeof(position_t), cudaMemcpyDeviceToHost);
			MonteRayMemcpy(rhs->tally, tally, num*sizeof(tally_t), cudaMemcpyDeviceToHost);
		}

		nAllocated = rhs->nAllocated;
		nUsed = rhs->nUsed;
	}

	CUDAHOST_CALLABLE_MEMBER unsigned add( position_t xarg, position_t yarg, position_t zarg) {

		MONTERAY_VERIFY( nUsed < nAllocated, "MonteRayNextEventEstimator::add -- Detector list is full.  Can't add any more detectors." );

		x[nUsed] = xarg;
		y[nUsed] = yarg;
		z[nUsed] = zarg;
		return nUsed++;
	}

	CUDA_CALLABLE_MEMBER unsigned size(void) const { return nUsed; }
	CUDA_CALLABLE_MEMBER unsigned capacity(void) const { return nAllocated; }

	CUDAHOST_CALLABLE_MEMBER void setExclusionRadius(position_t r) { radius = r; }
	CUDA_CALLABLE_MEMBER position_t getExclusionRadius(void) const { return radius; }

	CUDA_CALLABLE_MEMBER position_t getX(unsigned i) const { MONTERAY_ASSERT(i<nUsed); return x[i]; }
	CUDA_CALLABLE_MEMBER position_t getY(unsigned i) const { MONTERAY_ASSERT(i<nUsed); return y[i]; }
	CUDA_CALLABLE_MEMBER position_t getZ(unsigned i) const { MONTERAY_ASSERT(i<nUsed); return z[i]; }

	CUDA_CALLABLE_MEMBER position_t distance(unsigned i, position_t x,  position_t y, position_t z ) const {
		using namespace std;
		MONTERAY_ASSERT(i<nUsed);
		return sqrt( pow(getX(i) - x,2) +
					 pow(getY(i) - y,2) +
				     pow(getZ(i) - z,2)
				);
	}

	CUDA_CALLABLE_MEMBER position_t getDistanceDirection(
			unsigned i,
			position_t  x, position_t  y, position_t  z,
			position_t& u, position_t& v, position_t& w
			) const
	{
		using namespace std;
		MONTERAY_ASSERT(i<nUsed);
		u = getX(i) - x;
		v = getY(i) - y;
		w = getZ(i) - z;

		position_t dist = distance(i,x,y,z);
		position_t invDistance = 1/ dist;
		u *= invDistance;
		v *= invDistance;
		w *= invDistance;

		return dist;
	}

	template<unsigned N>
	CUDA_CALLABLE_MEMBER tally_t calcScore(
			DetectorIndex_t detectorIndex,
			position_t x, position_t y, position_t z,
			position_t u, position_t v, position_t w,
			gpuFloatType_t energies[N], gpuFloatType_t weights[N],
			unsigned locationIndex,
			ParticleType_t particleType
			)
	{
		const bool debug = true;

		MONTERAY_ASSERT(detectorIndex<nUsed);
		tally_t score = 0.0;

		int cells[2*MAXNUMVERTICES];
		gpuFloatType_t crossingDistances[2*MAXNUMVERTICES];

		unsigned numberOfCells;

		gpuFloatType_t dist = getDistanceDirection(
							  detectorIndex,
							  x, y, z,
							  u, v, w );

		float3_t pos = make_float3( x, y, z);
		float3_t dir = make_float3( u, v, w);

		numberOfCells = cudaRayTrace( pGridBins, cells, crossingDistances, pos, dir, dist, false);

		for( unsigned energyIndex=0; energyIndex < N; ++energyIndex) {
			gpuFloatType_t energy = energies[energyIndex];
			gpuFloatType_t weight = weights[energyIndex];

			tally_t partialScore = 0.0;
			unsigned HashBin = getHashBin(pHash, energy);

			if( debug ) {
				printf("Debug: MonteRayNextEventEstimator::calcScore -- energyIndex=%d, energy=%f, hashIndex=%d, weight=%f\n", energyIndex, energy, HashBin, weight);
			}
			gpuFloatType_t materialXS[MAXNUMMATERIALS];
			for( unsigned i=0; i < pMatList->numMaterials; ++i ){
				materialXS[i] = getTotalXS( pMatList, i, pHash, HashBin, energy, 1.0);
				if( debug ) {
					printf("Debug: MonteRayNextEventEstimator::calcScore -- materialIndex=%d, materialXS=%f\n", i, materialXS[i]);
				}
			}

			tally_t opticalThickness = 0.0;
			for( unsigned i=0; i < numberOfCells; ++i ){
				int cell = cells[i];
				gpuFloatType_t cellDistance = crossingDistances[i];
				if( cell == UINT_MAX ) continue;

				gpuFloatType_t totalXS = 0.0;
				unsigned numMaterials = getNumMats( pMatProps, cell);
				if( debug ) {
					printf("Debug: MonteRayNextEventEstimator::calcScore -- cell=%d, cellDistance=%f, numMaterials=%d\n", cell, cellDistance, numMaterials);
				}
				for( unsigned matIndex=0; matIndex<numMaterials; ++matIndex ) {
					unsigned matID = getMatID(pMatProps, cell, matIndex);
					gpuFloatType_t density = getDensity(pMatProps, cell, matIndex );
					gpuFloatType_t xs = materialXS[matID]*density;
					totalXS += xs;
					if( debug ) {
						printf("Debug: MonteRayNextEventEstimator::calcScore -- materialID=%d, density=%f, materialXS=%f, xs=%f, totalxs=%f\n", matID, materialXS[matID], density, xs, totalXS);
					}
				}

				opticalThickness += totalXS * cellDistance;
			}

			partialScore = ( weight / (2.0 * MonteRay::pi * dist*dist)  ) * exp( - opticalThickness);
			score += partialScore;
		}

		return score;
	}

	CUDAHOST_CALLABLE_MEMBER void setGeometry(const GridBinsHost* pGrid, const MonteRay_MaterialProperties* pMPs) {
		pGridBinsHost = pGrid;
		pGridBins = pGrid->getPtr();
		pMatPropsHost = pMPs;
		pMatProps = pMPs->getPtr();
	}

	CUDAHOST_CALLABLE_MEMBER void setMaterialList(const MonteRayMaterialListHost* ptr) {
		pMatListHost = ptr;
		pMatList = ptr->getPtr();
		pHashHost = ptr->getHashPtr();
		pHash = pHashHost->getPtr();
	}

private:
	unsigned nUsed;
	unsigned nAllocated;
	position_t radius;

	position_t* x;
	position_t* y;
	position_t* z;

	tally_t* tally;

	const GridBinsHost* pGridBinsHost;
	const GridBins* pGridBins;
	const MonteRay_MaterialProperties* pMatPropsHost;
	const MonteRay_MaterialProperties_Data* pMatProps;
	const MonteRayMaterialListHost* pMatListHost;
	const MonteRayMaterialList* pMatList;
	const HashLookupHost* pHashHost;
	const HashLookup* pHash;
};

} /* namespace MonteRay */

#endif /* MONTERAYNEXTEVENTESTIMATOR_HH_ */
