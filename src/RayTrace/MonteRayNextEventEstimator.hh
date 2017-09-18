#ifndef MONTERAYNEXTEVENTESTIMATOR_HH_
#define MONTERAYNEXTEVENTESTIMATOR_HH_

#include <iostream>

#include "MonteRayDefinitions.hh"
#include "MonteRayCopyMemory.hh"
#include "GPUErrorCheck.hh"
#include "GPUAtomicAdd.hh"
#include "RayList.hh"

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

	CUDAHOST_CALLABLE_MEMBER std::string className(){ return std::string("MonteRayNextEventEstimator");}

	CUDAHOST_CALLABLE_MEMBER MonteRayNextEventEstimator(unsigned num){
		if( num == 0 ) { num = 1; }
		if( Base::debug ) {
			std::cout << "RayList_t::RayList_t(n), n=" << num << " \n";
		}
		init();
		x = (position_t*) MONTERAYHOSTALLOC( num*sizeof( position_t ), Base::isManagedMemory, "x" );
		y = (position_t*) MONTERAYHOSTALLOC( num*sizeof( position_t ), Base::isManagedMemory, "y" );
		z = (position_t*) MONTERAYHOSTALLOC( num*sizeof( position_t ), Base::isManagedMemory, "z" );
		if( Base::debug ) {
			std::cout << "RayList_t::RayList_t -- allocating host tally memory. \n";
		}
		tally = (tally_t*) MONTERAYHOSTALLOC( num*sizeof( tally_t ), Base::isManagedMemory, "tally" );
		nAllocated = num;

		for( auto i=0; i<num; ++i){
			tally[i] = 0.0;
		}
	}

	CUDAHOST_CALLABLE_MEMBER ~MonteRayNextEventEstimator(){
		if( Base::isCudaIntermediate ) {
			if( x != NULL )	{ MonteRayDeviceFree( x ); }
			if( y != NULL )	{ MonteRayDeviceFree( y ); }
			if( z != NULL )	{ MonteRayDeviceFree( z ); }
			if( tally != NULL )	{ MonteRayDeviceFree( tally ); }
		} else {
			if( x != NULL )	{ MonteRayHostFree( x, Base::isManagedMemory ); }
			if( y != NULL )	{ MonteRayHostFree( y, Base::isManagedMemory ); }
			if( z != NULL )	{ MonteRayHostFree( z, Base::isManagedMemory ); }
			if( tally != NULL )	{ MonteRayHostFree( tally, Base::isManagedMemory ); }
		}
	}

	CUDAHOST_CALLABLE_MEMBER void init() {
		nUsed = 0;
		nAllocated = 0;
		radius = 0.0;
		x = NULL;
		y = NULL;
		z = NULL;
		tally = NULL;

		pGridBinsHost = NULL;
		pGridBins = NULL;
		pMatPropsHost = NULL;
		pMatProps = NULL;
		pMatListHost = NULL;
		pMatList = NULL;
		pHash = NULL;
		pHashHost = NULL;
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
				x = (position_t*) MONTERAYDEVICEALLOC( num*sizeof( position_t ), "x" );
			}
			if( y == NULL )	{
				y = (position_t*) MONTERAYDEVICEALLOC( num*sizeof( position_t ), "y" );
			}
			if( z == NULL )	{
				z = (position_t*) MONTERAYDEVICEALLOC( num*sizeof( position_t ), "z" );
			}
			if( tally == NULL )	{
				tally = (tally_t*) MONTERAYDEVICEALLOC( num*sizeof( tally_t ), "tally" );
			}

			MonteRayMemcpy(x, rhs->x, num*sizeof(position_t), cudaMemcpyHostToDevice);
			MonteRayMemcpy(y, rhs->y, num*sizeof(position_t), cudaMemcpyHostToDevice);
			MonteRayMemcpy(z, rhs->z, num*sizeof(position_t), cudaMemcpyHostToDevice);
			MonteRayMemcpy( tally, rhs->tally, num*sizeof(tally_t), cudaMemcpyHostToDevice);

			pGridBins = rhs->pGridBinsHost->getPtrDevice();
			pMatProps = rhs->pMatPropsHost->ptrData_device;
			pMatList = rhs->pMatListHost->ptr_device;
			pHash = rhs->pMatListHost->getHashPtr()->getPtrDevice();
		} else {
			// target is the host, origin is the intermediate

			if( Base::debug ) std::cout << "Debug: MonteRayNextEventEstimator::copy - copying tally from device to host\n";
			MonteRayMemcpy( tally, rhs->tally, num*sizeof(tally_t), cudaMemcpyDeviceToHost);
			if( Base::debug ) std::cout << "Debug: MonteRayNextEventEstimator::copy - DONE copying tally from device to host\n";
		}

		nAllocated = rhs->nAllocated;
		nUsed = rhs->nUsed;

		if( Base::debug ) {
			std::cout << "Debug: MonteRayNextEventEstimator::copy -- exitting." << std::endl;
		}
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

	CUDA_CALLABLE_MEMBER tally_t getTally(unsigned i) const { MONTERAY_ASSERT(i<nUsed); return tally[i]; }

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
		const bool debug = false;

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

		gpuFloatType_t logEnergies[N];
		for( unsigned energyIndex=0; energyIndex < N; ++energyIndex) {
			logEnergies[energyIndex] = log ( energies[energyIndex] );
		}
		for( unsigned energyIndex=0; energyIndex < N; ++energyIndex) {
			gpuFloatType_t logEnergy = logEnergies[energyIndex];
			gpuFloatType_t weight = weights[energyIndex];

			tally_t partialScore = 0.0;

			if( debug ) {
				printf("Debug: MonteRayNextEventEstimator::calcScore -- energyIndex=%d, energy=%f, weight=%f\n", energyIndex, std::exp( logEnergy ),  weight);
			}
			gpuFloatType_t materialXS[MAXNUMMATERIALS];
			for( unsigned i=0; i < pMatList->numMaterials; ++i ){
				if( debug ) printf("Debug: MonteRayNextEventEstimator::calcScore -- materialIndex=%d\n", i);
				materialXS[i] = getTotalXS( pMatList, i, logEnergy, 1.0);
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

	template<unsigned N>
	CUDA_CALLABLE_MEMBER void score( const RayList_t<N>* pRayList, unsigned tid ) {
		const bool debug = false;

		if( debug ) {
			printf("Debug: MonteRayNextEventEstimator::score -- tid=%d, particle=%d .\n",
					tid,  pRayList->points[tid].particleType);
		}

		// Neutrons are not yet supported
		if( pRayList->points[tid].particleType == neutron ) return;

		tally_t value = calcScore<N>( pRayList->points[tid].detectorIndex,
								   pRayList->points[tid].pos[0],
								   pRayList->points[tid].pos[1],
								   pRayList->points[tid].pos[2],
								   pRayList->points[tid].dir[0],
								   pRayList->points[tid].dir[1],
								   pRayList->points[tid].dir[2],
								   pRayList->points[tid].energy,
								   pRayList->points[tid].weight,
								   pRayList->points[tid].index,
								   pRayList->points[tid].particleType);
		if( debug ) {
			printf("Debug: MonteRayNextEventEstimator::score -- value=%f.\n" , value);
		}
		gpu_atomicAdd( &tally[pRayList->points[tid].detectorIndex], value );
	}

	template<unsigned N>
	void cpuScoreRayList( const RayList_t<N>* pRayList ) {
		for( auto i=0; i<pRayList->size(); ++i ) {
			score(pRayList,i);
		}
	}

#ifdef __CUDACC__
	template<unsigned N>
	void launch_ScoreRayList( unsigned nBlocks, unsigned nThreads, cudaStream_t stream, const RayList_t<N>* pRayList );
#endif

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

#ifdef __CUDACC__
template<unsigned N>
CUDA_CALLABLE_KERNEL void kernel_ScoreRayList(MonteRayNextEventEstimator* ptr, const RayList_t<N>* pRayList );
#endif

} /* namespace MonteRay */

#endif /* MONTERAYNEXTEVENTESTIMATOR_HH_ */
