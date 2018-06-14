#ifndef EXPECTEDPATHLENGTH_T_HH_
#define EXPECTEDPATHLENGTH_T_HH_

namespace MonteRay{

template<typename GRIDTYPE, unsigned N>
CUDA_CALLABLE_MEMBER void
tallyCollision(const GRIDTYPE* pGrid,
			   const MonteRayMaterialList* pMatList,
		       const MonteRay_MaterialProperties_Data* pMatProps,
		       const HashLookup* pHash,
		       const Ray_t<N>* p,
		       gpuTallyType_t* pTally,
		       unsigned tid )
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

	unsigned HashBin;
	if( p->particleType == neutron ) {
		HashBin = getHashBin(pHash, energy);
	}

	if( energy < 1e-20 ) {
		return;
	}

	int cells[2*MAXNUMVERTICES];
	gpuRayFloat_t crossingDistances[2*MAXNUMVERTICES];

	unsigned numberOfCells;

//	gpuRayFloat_t pos = make_float3( p->pos[0], p->pos[1], p->pos[2]);
//	gpuRayFloat_t dir = make_float3( p->dir[0], p->dir[1], p->dir[2]);
	Position_t pos( p->pos[0], p->pos[1], p->pos[2] );
	Direction_t dir( p->dir[0], p->dir[1], p->dir[2] );

	numberOfCells = pGrid->rayTrace( cells, crossingDistances, pos, dir, 1.0e6f, false);

	gpuFloatType_t materialXS[MAXNUMMATERIALS];
	for( unsigned i=0; i < pMatList->numMaterials; ++i ){
		if( p->particleType == neutron ) {
			materialXS[i] = getTotalXS( pMatList, i, pHash, HashBin, energy, 1.0);
		} else {
			materialXS[i] = getTotalXS( pMatList, i, energy, 1.0);
		}
	}

	for( unsigned i=0; i < numberOfCells; ++i ){
		int cell = cells[i];
		gpuRayFloat_t distance = crossingDistances[i];
		if( cell == UINT_MAX ) continue;

		opticalPathLength += tallyCellSegment(pMatList, pMatProps, materialXS, pTally,
				                              cell, distance, energy, p->weight[0], opticalPathLength);

		if( opticalPathLength > 5.0 ) {
			// cut off at 5 mean free paths
			return;
		}
	}
}

template<typename GRIDTYPE, unsigned N>
CUDA_CALLABLE_MEMBER
gpuTallyType_t
tallyAttenuation(GRIDTYPE* pGrid,
			     MonteRayMaterialList* pMatList,
			     MonteRay_MaterialProperties_Data* pMatProps,
			     const HashLookup* pHash,
			     Ray_t<N>* p){

	gpuTallyType_t enteringFraction = p->weight[0];
	gpuFloatType_t energy = p->energy[0];

	unsigned HashBin;
	if( p->particleType == neutron ) {
		HashBin = getHashBin(pHash, energy);
	}

	if( energy < 1e-20 ) {
		return 0.0;
	}

	int cells[2*MAXNUMVERTICES];
	gpuRayFloat_t crossingDistances[2*MAXNUMVERTICES];

	unsigned numberOfCells;

//	float3_t pos = make_float3( p->pos[0], p->pos[1], p->pos[2]);
//	float3_t dir = make_float3( p->dir[0], p->dir[1], p->dir[2]);
	MonteRay::Vector3D<gpuRayFloat_t> pos(p->pos[0], p->pos[1], p->pos[2]);
	MonteRay::Vector3D<gpuRayFloat_t> dir(p->dir[0], p->dir[1], p->dir[2]);

	numberOfCells = pGrid->rayTrace( cells, crossingDistances, pos, dir, 1.0e6, false);

	for( unsigned i=0; i < numberOfCells; ++i ){
		int cell = cells[i];
		gpuRayFloat_t distance = crossingDistances[i];
		if( cell == UINT_MAX ) continue;

		enteringFraction = attenuateRayTraceOnly(pMatList, pMatProps, pHash, HashBin, cell, distance, energy, enteringFraction, p->particleType );

		if( enteringFraction < 1e-11 ) {
			// cut off at 25 mean free paths
			return 0.0;
		}
	}
	return enteringFraction;
}

template<typename GRIDTYPE, unsigned N>
CUDA_CALLABLE_KERNEL void rayTraceTally(
		      const GRIDTYPE* pGrid,
			  const RayList_t<N>* pCP,
			  const MonteRayMaterialList* pMatList,
		      const MonteRay_MaterialProperties_Data* pMatProps,
		      const HashLookup* pHash,
		      gpuTallyType_t* tally){

	const bool debug = false;

#ifdef __CUDACC__
	unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
#else
	unsigned tid = 0;
#endif

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

		tallyCollision(pGrid, pMatList, pMatProps, pHash, &p, tally);

#ifdef __CUDACC__
		tid += blockDim.x*gridDim.x;
#else
		++tid;
#endif
	}
	return;
}

template<typename GRIDTYPE, unsigned N>
MonteRay::tripleTime launchRayTraceTally(
		                 std::function<void (void)> cpuWork,
		                 unsigned nBlocks,
		                 unsigned nThreads,
		                 const GRIDTYPE* pGrid,
		                 const RayListInterface<N>* pCP,
		                 const MonteRayMaterialListHost* pMatList,
		                 const MonteRay_MaterialProperties* pMatProps,
		                       gpuTallyHost* pTally
		                )
{
	MonteRay::tripleTime time;

#ifdef __CUDACC__
	cudaEvent_t startGPU, stopGPU, start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);

	cudaStream_t stream;
	cudaStreamCreate( &stream );

	cudaEventRecord(start,0);
	cudaEventRecord(startGPU,stream);

	rayTraceTally<<<nBlocks,nThreads,0,stream>>>(pGrid->getDevicePtr(), pCP->getPtrPoints()->devicePtr,
			pMatList->ptr_device, pMatProps->ptrData_device, pMatList->getHashPtr()->getPtrDevice(),
			pTally->temp->tally );

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
#else
	MonteRay::cpuTimer timer1, timer2;
	timer1.start();

	rayTraceTally( pGrid->getPtr(),
			       pCP->getPtrPoints(),
			   	   pMatList->getPtr(),
			   	   pMatProps->getPtr(),
			   	   pMatList->getHashPtr()->getPtr(),
				   pTally->getPtr()->tally
				 );
	timer1.stop();
	timer2.start();
	cpuWork();
	timer2.stop();

	time.gpuTime = timer1.getTime();
	time.cpuTime = timer2.getTime();
	time.totalTime = timer1.getTime() + timer2.getTime();
#endif


	return time;
}

} /* end namespace */

#endif /* EXPECTEDPATHLENGTH_T_HH_ */
