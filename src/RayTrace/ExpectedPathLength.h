#ifndef EXPECTEDPATHLENGTH_H_
#define EXPECTEDPATHLENGTH_H_


#include "CollisionPoints.h"
#include "SimpleMaterialList.h"
#include "SimpleMaterialProperties.h"
#include "gpuRayTrace.h"

#ifdef CUDA
__device__
void
tallyCollision(GridBins* pGrid, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuParticle_t* p, gpuFloatType_t* tally);

__device__
gpuFloatType_t
tallyCellSegment(SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuFloatType_t* tally, unsigned cell, double distance, double energy, double enteringFraction);

__device__
gpuFloatType_t
tallyAttenuation(GridBins* pGrid, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuParticle_t* p);

__device__
gpuFloatType_t
attenuateRayTraceOnly(SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, unsigned cell, double distance, double energy, double enteringFraction );

__global__
void
rayTraceTally(GridBins* pGrid, CollisionPoints* pCP, SimpleMaterialList* pMatList, SimpleMaterialProperties* pMatProps, gpuFloatType_t* tally);
#endif

#endif /* EXPECTEDPATHLENGTH_H_ */
