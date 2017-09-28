#include "MonteRayNextEventEstimator.hh"

namespace MonteRay {

#ifdef __CUDACC__
template<unsigned N>
CUDA_CALLABLE_KERNEL void kernel_ScoreRayList(MonteRayNextEventEstimator* ptr, const RayList_t<N>* pRayList ) {
	const bool debug = false;

	if( debug ) {
		printf("Debug: MonteRayNextEventEstimator::kernel_ScoreRayList\n");
	}

	unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;

	unsigned num = pRayList->size();
	while( tid < num ) {
		if( debug ) {
			printf("Debug: MonteRayNextEventEstimator::kernel_ScoreRayList -- tid=%d\n", tid);
		}
		ptr->score(pRayList,tid);
		tid += blockDim.x*gridDim.x;
	}
}
#endif

#ifdef __CUDACC__
template<unsigned N>
void
MonteRayNextEventEstimator::launch_ScoreRayList( unsigned nBlocks, unsigned nThreads, cudaStream_t& stream, const RayList_t<N>* pRayList ) {
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
	kernel_ScoreRayList<<<nBlocks, nThreads, 0, stream>>>( devicePtr, pRayList->devicePtr );
	if( debug ) {
		cudaError_t cudaerr = cudaDeviceSynchronize();
		if( cudaerr != cudaSuccess ) {
			printf("MonteRayNextEventEstimator::launch_ScoreRayList -- kernel_ScoreRayList launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
		}
	}
}

template void MonteRayNextEventEstimator::launch_ScoreRayList<1>( unsigned nBlocks, unsigned nThreads, cudaStream_t& stream, const RayList_t<1>* pRayList );
template void MonteRayNextEventEstimator::launch_ScoreRayList<3>( unsigned nBlocks, unsigned nThreads, cudaStream_t& stream, const RayList_t<3>* pRayList );
#endif

}


