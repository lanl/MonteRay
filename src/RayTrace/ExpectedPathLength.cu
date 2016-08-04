#include "ExpectedPathLength.h"
#include "GridBins.h"
#include "gpuTiming.h"
#include "gpuGlobal.h"

namespace MonteRay{

 __device__
 gpuFloatType_t
 tallyAttenuation(GridBins* pGrid, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuParticle_t* p){

	 gpuFloatType_t enteringFraction = p->weight;
	 gpuFloatType_t energy = p->energy;

	 if( energy < 1e-20 ) {
		 return enteringFraction;
	 }

	 int cells[2*MAXNUMVERTICES];
	 float_t crossingDistances[2*MAXNUMVERTICES];

	 unsigned numberOfCells;

	 float3_t pos = make_float3( p->pos.x, p->pos.y, p->pos.z);
	 float3_t dir = make_float3( p->dir.u, p->dir.v, p->dir.w);

	 numberOfCells = cudaRayTrace( pGrid, cells, crossingDistances, pos, dir, 1.0e6f, false);



	 for( unsigned i=0; i < numberOfCells; ++i ){
		 int cell = cells[i];
		 gpuFloatType_t distance = crossingDistances[i];
		 if( cell == UINT_MAX ) continue;

		 enteringFraction = attenuateRayTraceOnly(pMatList, pMatProps, cell, distance, energy, enteringFraction );

		 if( enteringFraction < 1e-11 ) {
			 // cut off at 25 mean free paths
			 return enteringFraction;
		 }
	 }
	 return enteringFraction;
 }

 __device__
 gpuFloatType_t
 attenuateRayTraceOnly(SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, unsigned cell, gpuFloatType_t distance, gpuFloatType_t energy, gpuFloatType_t enteringFraction ) {

	 gpuFloatType_t totalXS = 0.0;
	 unsigned numMaterials = getNumMats( pMatProps, cell);
	 for( unsigned i=0; i<numMaterials; ++i ) {

		 unsigned matID = getMatID(pMatProps, cell, i);
		 gpuFloatType_t density = getDensity(pMatProps, cell, i );
		 if( density > 1e-5 ) {
			 totalXS +=  getTotalXS( pMatList, matID, energy, density);
		 }
	 }

	 gpuFloatType_t attenuation = 1.0;

	 if( totalXS > 1e-5 ) {
		 attenuation = expf( - totalXS*distance );
	 }
	 return enteringFraction * attenuation;

 }


__device__ void tallyCollision(GridBins* pGrid, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuParticle_t* p, gpuTallyType_t* tally){

	 gpuFloatType_t enteringFraction = p->weight;
	 gpuFloatType_t energy = p->energy;

	 if( energy < 1e-20 ) {
		 return;
	 }

	int cells[2*MAXNUMVERTICES];
	float_t crossingDistances[2*MAXNUMVERTICES];

	unsigned numberOfCells;

	float3_t pos = make_float3( p->pos.x, p->pos.y, p->pos.z);
	float3_t dir = make_float3( p->dir.u, p->dir.v, p->dir.w);

	numberOfCells = cudaRayTrace( pGrid, cells, crossingDistances, pos, dir, 1.0e6f, false);

	for( unsigned i=0; i < numberOfCells; ++i ){
		int cell = cells[i];
		gpuFloatType_t distance = crossingDistances[i];
		if( cell == UINT_MAX ) continue;

		enteringFraction = tallyCellSegment(pMatList, pMatProps, tally, cell, distance, energy, enteringFraction);

		if( enteringFraction < 1e-11 ) {
			// cut off at 25 mean free paths
			return;
		}
	}
}

__device__
gpuFloatType_t
tallyCellSegment(SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuTallyType_t* tally, unsigned cell, gpuFloatType_t distance, gpuFloatType_t energy, gpuFloatType_t enteringFraction ) {

	gpuFloatType_t totalXS = 0.0;
     unsigned numMaterials = getNumMats( pMatProps, cell);
     for( unsigned i=0; i<numMaterials; ++i ) {

         unsigned matID = getMatID(pMatProps, cell, i);
         gpuFloatType_t density = getDensity(pMatProps, cell, i );
         if( density > 1e-5 ) {
             totalXS +=  getTotalXS( pMatList, matID, energy, density);
         }
     }

     gpuFloatType_t attenuation = 1.0;
     gpuFloatType_t score = distance;
     if( totalXS > 1e-5 ) {
         attenuation = expf( - totalXS*distance );
         if(  (1 - attenuation) > 1e-5 ) {
             score = ( 1.0 / totalXS ) * ( 1.0 - attenuation );
         }
     }
     score *= enteringFraction;

     //atomicAddDouble( &tally[cell], score);
     atomicAdd( &tally[cell], score);

     return enteringFraction * attenuation;
}

__global__
void rayTraceTally(GridBins* pGrid, CollisionPoints* pCP, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuTallyType_t* tally){

	unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned N = pCP->size;

	while( tid < N ) {
		gpuParticle_t p = getParticle( pCP, tid);
		tallyCollision(pGrid, pMatList, pMatProps, &p, tally);

		tid += blockDim.x*gridDim.x;
	}
	return;
}

__device__ void tallyCollision(GridBins* pGrid, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuParticle_t* p, gpuTally* pTally, unsigned tid){
	bool debug = false;

	if( debug ) {
		printf("--------------------------------------------------------------------------------------------------------\n");
		printf("GPU::tallyCollision:: nCollisions=%d, x=%f, y=%f, z=%f, u=%f, v=%f, w=%f, weight=%f, index=%d \n",
				tid+1,
				p->pos.x,
				p->pos.y,
				p->pos.z,
				p->dir.u,
				p->dir.v,
				p->dir.w,
				p->weight,
				p->index
		);
	}

	typedef double enteringFraction_t;

	enteringFraction_t enteringFraction = p->weight;
	gpuFloatType_t energy = p->energy;

	if( energy < 1e-20 ) {
		return;
	}

	int cells[2*MAXNUMVERTICES];
	float_t crossingDistances[2*MAXNUMVERTICES];

	unsigned numberOfCells;

	float3_t pos = make_float3( p->pos.x, p->pos.y, p->pos.z);
	float3_t dir = make_float3( p->dir.u, p->dir.v, p->dir.w);

	numberOfCells = cudaRayTrace( pGrid, cells, crossingDistances, pos, dir, 1.0e6f, false);

	for( unsigned i=0; i < numberOfCells; ++i ){
		int cell = cells[i];
		gpuFloatType_t distance = crossingDistances[i];
		if( cell == UINT_MAX ) continue;

		enteringFraction = tallyCellSegment(pMatList, pMatProps, pTally, cell, distance, energy, enteringFraction);

		if( enteringFraction < 1e-11 ) {
			// cut off at 25 mean free paths
			return;
		}
	}
}

__device__
double
tallyCellSegment(SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuTally* pTally, unsigned cell, gpuFloatType_t distance, gpuFloatType_t energy, double enteringFraction ) {
	bool debug = false;

	typedef double xs_t;
	typedef double attenuation_t;
	typedef double score_t;

	xs_t totalXS = 0.0;
	unsigned numMaterials = getNumMats( pMatProps, cell);
	if( debug ) {
		printf("GPU::tallyCellSegment:: cell=%d, numMaterials=%d\n", cell, numMaterials);
	}
	for( unsigned i=0; i<numMaterials; ++i ) {
		unsigned matID = getMatID(pMatProps, cell, i);
		gpuFloatType_t density = getDensity(pMatProps, cell, i );
		xs_t xs = 0.0;
		if( density > 1e-5 ) {
			xs =  getTotalXS( pMatList, matID, energy, density);
			totalXS += xs;
		}
		if( debug ) {
			printf("GPU::tallyCellSegment::       material=%d, density=%f, xs=%f, totalxs=%f\n", i, density, xs, totalXS);
		}
	}

	attenuation_t attenuation = 1.0;
	score_t score = distance;
	if( totalXS > 1e-5 ) {
		attenuation = exp( - totalXS*distance );
		if(  (1.0 - attenuation) > 1e-5 ) {
			score = ( 1.0 / totalXS ) * ( 1.0 - attenuation );
		}
	}
	score *= enteringFraction;

	if( debug ) {
		printf("GPU::tallyCellSegment:: cell=%d, distance=%f, totalXS=%f, score=%f\n", cell, distance, totalXS, score);
	}

	//atomicAdd( &(tally->tally[cell]), score);
	MonteRay::score( pTally, cell, score );

	if( debug ) {
		printf("GPU::tallyCellSegment:: total score=%f\n", pTally->tally[cell] );
	}

	return (enteringFraction * attenuation);
}


__global__
void rayTraceTally(GridBins* pGrid, CollisionPoints* pCP, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuTally* pTally ){

	unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned N = pCP->size;

	while( tid < N ) {
//		printf("--------------------------------------------------------------------------------------------------------\n");
//	    printf("GPU::rayTraceTally:: tid=%d\n", tid );

		gpuParticle_t p = getParticle( pCP, tid);

		tallyCollision(pGrid, pMatList, pMatProps, &p, pTally, tid);

		tid += blockDim.x*gridDim.x;
	}
	return;
}

MonteRay::tripleTime launchRayTraceTally(
		                 std::function<void (void)> cpuWork,
		                 unsigned nBlocks,
		                 unsigned nThreads,
		                 GridBinsHost* pGrid,
		                 CollisionPointsHost* pCP,
		                 SimpleMaterialListHost* pMatList,
		                 SimpleMaterialPropertiesHost* pMatProps,
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

	rayTraceTally<<<nBlocks,nThreads,0,stream>>>(pGrid->ptr_device, pCP->ptrPoints_device, pMatList->ptr_device, pMatProps->ptr_device, pTally->ptr_device);
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

	float totalTime;
	cudaEventElapsedTime(&totalTime, start, stop );
	time.totalTime = totalTime/1000.0;

	return time;
}

}
