#include "MonteRayNextEventEstimator.hh"

namespace MonteRay {

#ifdef __CUDACC__
template<unsigned N>
CUDA_CALLABLE_KERNEL void kernel_ScoreRayList(MonteRayNextEventEstimator* ptr, const RayList_t<N>* pRayList ) {
	unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;

	unsigned num = pRayList->size();
	while( tid < num ) {
		ptr->score(pRayList,tid);
		tid += blockDim.x*gridDim.x;
	}
}
#endif

#ifdef __CUDACC__
template<unsigned N>
void
MonteRayNextEventEstimator::launch_ScoreRayList( unsigned nBlocks, unsigned nThreads, cudaStream_t stream, const RayList_t<N>* pRayList ) {
	kernel_ScoreRayList<<<nBlocks, nThreads, 0, stream>>>( devicePtr, pRayList->devicePtr );
}

template void MonteRayNextEventEstimator::launch_ScoreRayList<1>( unsigned nBlocks, unsigned nThreads, cudaStream_t stream, const RayList_t<1>* pRayList );
template void MonteRayNextEventEstimator::launch_ScoreRayList<3>( unsigned nBlocks, unsigned nThreads, cudaStream_t stream, const RayList_t<3>* pRayList );
#endif

}


