#include "ExpectedPathLength.h"

#include <math.h>

#include "GridBins.h"
#include "GPUTiming.hh"
#include "MonteRayDefinitions.hh"
#include "GPUAtomicAdd.hh"

namespace MonteRay{

 template<unsigned N>
 __device__
 gpuTallyType_t
 tallyAttenuation(GridBins* pGrid, SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, Ray_t<N>* p){


	 gpuTallyType_t enteringFraction = p->weight[0];
	 gpuFloatType_t energy = p->energy[0];
	 unsigned HashBin = getHashBin(pHash, energy);

	 if( energy < 1e-20 ) {
		 return enteringFraction;
	 }

	 int cells[2*MAXNUMVERTICES];
	 gpuFloatType_t crossingDistances[2*MAXNUMVERTICES];

	 unsigned numberOfCells;

	 float3_t pos = make_float3( p->pos[0], p->pos[1], p->pos[2]);
	 float3_t dir = make_float3( p->dir[0], p->dir[1], p->dir[2]);

	 numberOfCells = cudaRayTrace( pGrid, cells, crossingDistances, pos, dir, 1.0e6f, false);

	 for( unsigned i=0; i < numberOfCells; ++i ){
		 int cell = cells[i];
		 gpuFloatType_t distance = crossingDistances[i];
		 if( cell == UINT_MAX ) continue;

		 enteringFraction = attenuateRayTraceOnly(pMatList, pMatProps, pHash, HashBin, cell, distance, energy, enteringFraction );

		 if( enteringFraction < 1e-11 ) {
			 // cut off at 25 mean free paths
			 return enteringFraction;
		 }
	 }
	 return enteringFraction;
 }

 template __device__ gpuTallyType_t
 tallyAttenuation<1>(GridBins* pGrid, SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, Ray_t<1>* p);

 template __device__ gpuTallyType_t
 tallyAttenuation<3>(GridBins* pGrid, SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, Ray_t<3>* p);


 __device__
 gpuTallyType_t
 attenuateRayTraceOnly(SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, unsigned HashBin, unsigned cell, gpuFloatType_t distance, gpuFloatType_t energy, gpuTallyType_t enteringFraction ) {

	 gpuTallyType_t totalXS = 0.0;
	 unsigned numMaterials = getNumMats( pMatProps, cell);
	 for( unsigned i=0; i<numMaterials; ++i ) {

		 unsigned matID = getMatID(pMatProps, cell, i);
		 gpuFloatType_t density = getDensity(pMatProps, cell, i );
		 if( density > 1e-5 ) {
			 //unsigned materialIndex = materialIDtoIndex(pMatList, matID);
			 totalXS +=  getTotalXS( pMatList, matID, pHash, HashBin, energy, density);
		 }
	 }

	 gpuTallyType_t attenuation = 1.0;

	 if( totalXS > 1e-5 ) {
		 attenuation = exp( - totalXS*distance );
	 }
	 return enteringFraction * attenuation;

 }

 template<unsigned N> __device__ void
 tallyCollision(GridBins* pGrid, SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, Ray_t<N>* p, gpuTallyType_t* tally){

	 gpuTallyType_t opticalPathLength = 0.0;
	 gpuFloatType_t energy = p->energy[0];
	 unsigned HashBin = getHashBin(pHash, energy);

	 if( energy < 1e-20 ) {
		 return;
	 }

	int cells[2*MAXNUMVERTICES];
	gpuFloatType_t crossingDistances[2*MAXNUMVERTICES];

	unsigned numberOfCells;

	float3_t pos = make_float3( p->pos[0], p->pos[1], p->pos[2]);
	float3_t dir = make_float3( p->dir[0], p->dir[1], p->dir[2]);

	numberOfCells = cudaRayTrace( pGrid, cells, crossingDistances, pos, dir, 1.0e6f, false);

	gpuFloatType_t materialXS[MAXNUMMATERIALS];
	for( unsigned i=0; i < pMatList->numMaterials; ++i ){
		materialXS[i] = getTotalXS( pMatList, i, pHash, HashBin, energy, 1.0);
	}

	for( unsigned i=0; i < numberOfCells; ++i ){
		int cell = cells[i];
		gpuFloatType_t distance = crossingDistances[i];
		if( cell == UINT_MAX ) continue;

		opticalPathLength += tallyCellSegment(pMatList, pMatProps, materialXS, tally, cell, distance, energy, p->weight[0], opticalPathLength );

		if( opticalPathLength > 5.0 ) {
			// cut off at 5 mean free paths
			return;
		}
	}
}

 template __device__ void
 tallyCollision<1>(GridBins* pGrid, SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, Ray_t<1>* p, gpuTallyType_t* tally);

 template __device__ void
 tallyCollision<3>(GridBins* pGrid, SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, Ray_t<3>* p, gpuTallyType_t* tally);

__device__
gpuTallyType_t
tallyCellSegment(SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps, gpuFloatType_t* materialXS, gpuTallyType_t* tally, unsigned cell, gpuFloatType_t distance, gpuFloatType_t energy, gpuFloatType_t weight, gpuTallyType_t opticalPathLength ) {

	gpuTallyType_t totalXS = 0.0;
	unsigned numMaterials = getNumMats( pMatProps, cell);
	for( unsigned i=0; i<numMaterials; ++i ) {

		unsigned matID = getMatID(pMatProps, cell, i);
		gpuFloatType_t density = getDensity(pMatProps, cell, i );
		if( density > 1e-5 ) {
	       	//unsigned materialIndex = materialIDtoIndex(pMatList, matID);
			//totalXS +=   getTotalXS( pMatList, matID, energy, density);
			totalXS +=   materialXS[matID]*density;
		}
	}

	gpuTallyType_t attenuation = 1.0;
	gpuTallyType_t score = distance;
	gpuTallyType_t cellOpticalPathLength = totalXS*distance;

	if( totalXS >  1e-5 ) {
		attenuation =  exp( - cellOpticalPathLength );
		score = ( 1.0 / totalXS ) * ( 1.0 - attenuation );
	}
	score *= exp( -opticalPathLength ) * weight;

	gpu_atomicAdd( &tally[cell], score);

	return cellOpticalPathLength;
}

template<unsigned N> __global__ void
rayTraceTally(GridBins* pGrid, RayList_t<N>* pCP, SimpleMaterialList* pMatList,
		      MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash,
		      gpuTallyType_t* tally){

	const bool debug = false;

	unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned num = pCP->size();

	if( debug ) printf("GPU::rayTraceTally:: starting tid=%d  N=%d\n", tid, N );

	while( tid < num ) {
		Ray_t<N> p = pCP->getParticle(tid);
		tallyCollision(pGrid, pMatList, pMatProps, pHash, &p, tally);

		tid += blockDim.x*gridDim.x;
	}
	return;
}

template __global__ void
rayTraceTally<1>(GridBins* pGrid, RayList_t<1>* pCP, SimpleMaterialList* pMatList,
		         MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash,
		         gpuTallyType_t* tally);

template __global__ void
rayTraceTally<3>(GridBins* pGrid, RayList_t<3>* pCP, SimpleMaterialList* pMatList,
		         MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash,
		         gpuTallyType_t* tally);

template<unsigned N> __device__ void
tallyCollision(GridBins* pGrid, SimpleMaterialList* pMatList,
		            MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, Ray_t<N>* p,
		            gpuTally* pTally, unsigned tid)
{
	const bool debug = false;

	if( debug ) {
		printf("--------------------------------------------------------------------------------------------------------\n");
		printf("GPU::tallyCollision:: nCollisions=%d, x=%f, y=%f, z=%f, u=%f, v=%f, e=%f, w=%f, weight=%f, index=%d \n",
				tid+1,
				p->pos[0],
				p->pos[1],
				p->pos[2],
				p->dir[0],
				p->dir[1],
				p->dir[2],
				p->energy[0],
				p->weight[0],
				p->index
		);
	}

	typedef gpuTallyType_t enteringFraction_t;

	gpuTallyType_t opticalPathLength = 0.0;

	gpuFloatType_t energy = p->energy[0];
	unsigned HashBin = getHashBin(pHash, energy);

	if( energy < 1e-20 ) {
		return;
	}

	int cells[2*MAXNUMVERTICES];
	gpuFloatType_t crossingDistances[2*MAXNUMVERTICES];

	unsigned numberOfCells;

	float3_t pos = make_float3( p->pos[0], p->pos[1], p->pos[2]);
	float3_t dir = make_float3( p->dir[0], p->dir[1], p->dir[2]);

	numberOfCells = cudaRayTrace( pGrid, cells, crossingDistances, pos, dir, 1.0e6f, false);

	gpuFloatType_t materialXS[MAXNUMMATERIALS];
	for( unsigned i=0; i < pMatList->numMaterials; ++i ){
		materialXS[i] = getTotalXS( pMatList, i, pHash, HashBin, energy, 1.0);
	}

	for( unsigned i=0; i < numberOfCells; ++i ){
		int cell = cells[i];
		gpuFloatType_t distance = crossingDistances[i];
		if( cell == UINT_MAX ) continue;

		opticalPathLength += tallyCellSegment(pMatList, pMatProps, materialXS, pTally,
				                              cell, distance, energy, p->weight[0], opticalPathLength);

		if( opticalPathLength > 5.0 ) {
			// cut off at 5 mean free paths
			return;
		}
	}
}

template __device__ void
tallyCollision<1>(GridBins* pGrid, SimpleMaterialList* pMatList,
		            MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, Ray_t<1>* p,
		            gpuTally* pTally, unsigned tid);

template __device__ void
tallyCollision<3>(GridBins* pGrid, SimpleMaterialList* pMatList,
		            MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, Ray_t<3>* p,
		            gpuTally* pTally, unsigned tid);

__device__ gpuTallyType_t
tallyCellSegment(SimpleMaterialList* pMatList, MonteRay_MaterialProperties_Data* pMatProps,
		         gpuFloatType_t* materialXS , struct gpuTally* pTally, unsigned cell,
		         gpuFloatType_t distance, gpuFloatType_t energy, gpuFloatType_t weight,
		         gpuTallyType_t opticalPathLength ) {
	const bool debug = false;

	typedef gpuTallyType_t xs_t;
	typedef gpuTallyType_t attenuation_t;
	typedef gpuTallyType_t score_t;

	xs_t totalXS = 0.0;
	unsigned numMaterials = getNumMats( pMatProps, cell);
	if( debug ) {
		printf("GPU::tallyCellSegment:: cell=%d, numMaterials=%d\n", cell, numMaterials);
	}
	for( unsigned i=0; i<numMaterials; ++i ) {
		unsigned matID = getMatID(pMatProps, cell, i);
		gpuFloatType_t density = getDensity(pMatProps, cell, i );
        if( density > 1e-5 ) {
//        	totalXS +=   getTotalXS( pMatList, matID, energy, density);
//             totalXS +=   getTotalXS( pMatList, matID, pHash, HashBin, energy, density);
        	//unsigned materialIndex = materialIDtoIndex(pMatList, matID);
            totalXS +=   materialXS[matID]*density;
        }
//		if( debug ) {
//			printf("GPU::tallyCellSegment::       material=%d, density=%f, xs=%f, totalxs=%f\n", i, density, xs, totalXS);
//		}
	}

	attenuation_t attenuation = 1.0;
	score_t score = distance;
	gpuTallyType_t cellOpticalPathLength = totalXS*distance;

	if( totalXS >  1e-5 ) {
		attenuation =  exp( - cellOpticalPathLength );
		score = ( 1.0 / totalXS ) * ( 1.0 - attenuation );
	}
	score *= exp( -opticalPathLength ) * weight;

	if( debug ) {
		printf("GPU::tallyCellSegment:: cell=%d, distance=%f, totalXS=%f, score=%f\n", cell, distance, totalXS, score);
	}

	//atomicAdd( &(tally->tally[cell]), score);
	MonteRay::score( pTally, cell, score );

	if( debug ) {
		printf("GPU::tallyCellSegment:: total score=%f\n", pTally->tally[cell] );
	}

	return cellOpticalPathLength;
}

template<unsigned N> __global__ void
rayTraceTally(GridBins* pGrid, RayList_t<N>* pCP, SimpleMaterialList* pMatList,
		      MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, gpuTally* pTally ){

	const bool debug = false;

	unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;

	unsigned num = pCP->size();

	if( debug ) printf("GPU::rayTraceTally:: starting tid=%d  N=%d\n", tid, N );

	while( tid < num ) {
		Ray_t<N> p = pCP->getParticle(tid);

		if( debug ) {
		    printf("--------------------------------------------------------------------------------------------------------\n");
            printf("GPU::rayTraceTally:: tid=%d\n", tid );
            printf("GPU::rayTraceTally:: x=%f\n", p.pos[0] );
            printf("GPU::rayTraceTally:: y=%f\n", p.pos[1] );
            printf("GPU::rayTraceTally:: z=%f\n", p.pos[2] );
            printf("GPU::rayTraceTally:: u=%f\n", p.dir[0] );
            printf("GPU::rayTraceTally:: v=%f\n", p.dir[1] );
            printf("GPU::rayTraceTally:: w=%f\n", p.dir[2] );
            printf("GPU::rayTraceTally:: energy=%f\n", p.energy[0] );
            printf("GPU::rayTraceTally:: weight=%f\n", p.weight[0] );
            printf("GPU::rayTraceTally:: index=%d\n", p.index );
		}

		tallyCollision(pGrid, pMatList, pMatProps, pHash, &p, pTally, tid);

		tid += blockDim.x*gridDim.x;
	}
	return;
}

template __global__ void
rayTraceTally<1>(GridBins* pGrid, RayList_t<1>* pCP, SimpleMaterialList* pMatList,
		         MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, gpuTally* pTally );

template __global__ void
rayTraceTally<3>(GridBins* pGrid, RayList_t<3>* pCP, SimpleMaterialList* pMatList,
		         MonteRay_MaterialProperties_Data* pMatProps, HashLookup* pHash, gpuTally* pTally );

template<unsigned N>
MonteRay::tripleTime launchRayTraceTally(
		                 std::function<void (void)> cpuWork,
		                 unsigned nBlocks,
		                 unsigned nThreads,
		                 GridBinsHost* pGrid,
		                 RayListInterface<N>* pCP,
		                 SimpleMaterialListHost* pMatList,
		                 MonteRay_MaterialProperties* pMatProps,
		                 gpuTallyHost* pTally
		                )
{
	MonteRay::tripleTime time;

	cudaEvent_t startGPU, stopGPU, start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);

	cudaStream_t stream;
	cudaStreamCreate( &stream );

	cudaEventRecord(start,0);
	cudaEventRecord(startGPU,stream);

	rayTraceTally<<<nBlocks,nThreads,0,stream>>>(pGrid->ptr_device, pCP->getPtrPoints()->devicePtr, pMatList->ptr_device, pMatProps->ptrData_device, pMatList->getHashPtr()->getPtrDevice(), pTally->ptr_device);
	cudaEventRecord(stopGPU,stream);
	cudaStreamWaitEvent(stream, stopGPU, 0);

	{
		MonteRay::cpuTimer timer;
		timer.start();
		cpuWork();
		timer.stop();
		time.cpuTime = timer.getTime();
	}

	cudaStreamSynchronize( stream );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaStreamDestroy(stream);

	float_t gpuTime;
	cudaEventElapsedTime(&gpuTime, startGPU, stopGPU );
	time.gpuTime = gpuTime / 1000.0;

	float_t totalTime;
	cudaEventElapsedTime(&totalTime, start, stop );
	time.totalTime = totalTime/1000.0;

	return time;
}

template MonteRay::tripleTime
launchRayTraceTally<1>( std::function<void (void)> cpuWork, unsigned nBlocks, unsigned nThreads,
		                GridBinsHost* pGrid, RayListInterface<1>* pCP, SimpleMaterialListHost* pMatList,
		                MonteRay_MaterialProperties* pMatProps, gpuTallyHost* pTally );

template MonteRay::tripleTime
launchRayTraceTally<3>( std::function<void (void)> cpuWork, unsigned nBlocks, unsigned nThreads,
		                GridBinsHost* pGrid, RayListInterface<3>* pCP, SimpleMaterialListHost* pMatList,
		                MonteRay_MaterialProperties* pMatProps, gpuTallyHost* pTally );

}


