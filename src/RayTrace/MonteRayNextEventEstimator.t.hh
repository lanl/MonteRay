#ifndef MONTERAYNEXTEVENTESTIMATOR_T_HH_
#define MONTERAYNEXTEVENTESTIMATOR_T_HH_

namespace MonteRay {

template<typename GRID_T, unsigned N>
CUDA_CALLABLE_KERNEL
void kernel_ScoreRayList(MonteRayNextEventEstimator<GRID_T>* ptr, const RayList_t<N>* pRayList ) {
	const bool debug = false;

	if( debug ) {
		printf("Debug: MonteRayNextEventEstimator::kernel_ScoreRayList\n");
	}

#ifdef __CUDACC__
	unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
#else
	unsigned tid = 0;
#endif

	unsigned num = pRayList->size();
	while( tid < num ) {
		if( debug ) {
			printf("Debug: MonteRayNextEventEstimator::kernel_ScoreRayList -- tid=%d\n", tid);
		}
		ptr->score(pRayList,tid);

#ifdef __CUDACC__
		tid += blockDim.x*gridDim.x;
#else
		++tid;
#endif
	}
}

template<typename GRID_T>
#ifdef __CUDACC__
template<unsigned N>
void MonteRayNextEventEstimator<GRID_T>::launch_ScoreRayList( unsigned nBlocks, unsigned nThreads, cudaStream_t& stream, const RayList_t<N>* pRayList )
#else
template<unsigned N>
void MonteRayNextEventEstimator<GRID_T>::launch_ScoreRayList( unsigned nBlocks, unsigned nThreads, const RayList_t<N>* pRayList )
#endif
	{
	const bool debug = false;

	const unsigned nRays = pRayList->size();
	if( nThreads > nRays ) {
		nThreads = nRays;
	}
	nThreads = (( nThreads + 32 -1 ) / 32 ) *32;

    const unsigned numThreadOverload = nBlocks;
    nBlocks = std::min(( nRays + numThreadOverload*nThreads -1 ) / (numThreadOverload*nThreads), 65535U);

	if( debug ) {
		printf("Debug: MonteRayNextEventEstimator::launch_ScoreRayList -- launching kernel_ScoreRayList with %d blocks, %d threads, to process %d rays\n", nBlocks, nThreads, nRays);
	}
#ifdef __CUDACC__
	kernel_ScoreRayList<<<nBlocks, nThreads, 0, stream>>>( Base::devicePtr, pRayList->devicePtr );
	if( debug ) {
		cudaError_t cudaerr = cudaDeviceSynchronize();
		if( cudaerr != cudaSuccess ) {
			printf("MonteRayNextEventEstimator::launch_ScoreRayList -- kernel_ScoreRayList launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
		}
	}
#else
	kernel_ScoreRayList( this, pRayList );
#endif

}

}

#endif /* MONTERAYNEXTEVENTESTIMATOR_T_HH_ */


