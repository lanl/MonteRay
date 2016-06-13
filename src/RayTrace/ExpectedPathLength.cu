#include "ExpectedPathLength.h"


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

		 if( enteringFraction < 1e-12 ) {
			 // cut off at 25 mean free paths
			 return enteringFraction;
		 }
	 }
	 return enteringFraction;
 }

 __device__
 gpuFloatType_t
 attenuateRayTraceOnly(SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, unsigned cell, double distance, double energy, double enteringFraction ) {

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


__device__ void tallyCollision(GridBins* pGrid, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuParticle_t* p, gpuFloatType_t* tally){

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

		if( enteringFraction < 1e-12 ) {
			// cut off at 25 mean free paths
			return;
		}
	}
}

__device__
gpuFloatType_t
tallyCellSegment(SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuFloatType_t* tally, unsigned cell, double distance, double energy, double enteringFraction ) {

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

     atomicAdd( &tally[cell], score);

     return enteringFraction * attenuation;
}

__global__
void rayTraceTally(GridBins* pGrid, CollisionPoints* pCP, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuFloatType_t* tally){

	unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned N = pCP->size;

	while( tid < N ) {
		gpuParticle_t p = getParticle( pCP, tid);
		tallyCollision(pGrid, pMatList, pMatProps, &p, tally);

		tid += blockDim.x*gridDim.x;
	}
	return;
}
