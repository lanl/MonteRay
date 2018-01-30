#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"
#include "GPUAtomicAdd.hh"
#include "ExpectedPathLength.hh"

#include "RayTraceNightly_genericGPU_test_helper.hh"

template<unsigned N>
FIGenericGPUTestHelper<N>::FIGenericGPUTestHelper(unsigned num){
}

template<unsigned N>
FIGenericGPUTestHelper<N>::~FIGenericGPUTestHelper(){
}

template<unsigned N>
void FIGenericGPUTestHelper<N>::setupTimers(){
#ifdef __CUDACC__
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#else
	timer.start();
#endif
}

template<unsigned N>
void FIGenericGPUTestHelper<N>::stopTimers(){
#ifdef __CUDACC__
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	gpuErrchk( cudaPeekAtLastError() );

	cudaEventElapsedTime(&elapsedTime, start, stop );

	std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;

	gpuErrchk( cudaPeekAtLastError() );
#else
	timer.stop();
	std::cout << "Elapsed time in non-CUDA kernel=" << timer.getTime()*1000.0 << " msec" << std::endl;
#endif
}

template class FIGenericGPUTestHelper<1>;
template class FIGenericGPUTestHelper<3>;

