#include "gpuTiming_test_helper.hh"

#include <iostream>
#include <ctime>
#include <iomanip>
#include <time.h>

#include "GPUTiming.hh"
#include "MonteRay_timer.hh"

#ifndef __CUDACC__
#include <cmath>
#endif

using namespace MonteRay;

GPUTimingTestHelper::GPUTimingTestHelper(){
}

GPUTimingTestHelper::~GPUTimingTestHelper(){
}

CUDA_CALLABLE_KERNEL  kernelGPUSleep(clock64_t nCycles, gpuTiming* timer) {
#ifdef __CUDACC__
	timer->start = clock64();
	clock64_t cycles = 0;
	clock64_t start = clock();
    while(cycles < nCycles) {
        cycles = clock64() - start;
    }
    timer->stop = clock64();
#else
	timer->start = clock();
	clock64_t cycles = 0;
	clock64_t start = clock();
    while(cycles < nCycles) {
        cycles = clock() - start;
    }
    timer->stop = clock();
#endif
}

void GPUTimingTestHelper::launchGPUSleep( clock64_t nCycles, MonteRay::gpuTimingHost* timer) {
#ifdef __CUDACC__
	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelGPUSleep<<<1,1>>>(nCycles, timer->ptr_device);
	gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);
#else
	kernelGPUSleep(nCycles, timer->getPtr() );
#endif

	return;
}


CUDA_CALLABLE_KERNEL  kernelGPUSleep(clock64_t nCycles) {
#ifdef __CUDACC__
	clock64_t cycles = 0;
	clock64_t start = clock64();
    while(cycles < nCycles) {
        cycles = clock64() - start;
    }
#else
	clock64_t cycles = 0;
	clock64_t start = clock();
    while(cycles < nCycles) {
        cycles = clock() - start;
    }
#endif
}

int frequency_of_primes (int n) {
  int i,j;
  int freq=n-1;
#ifdef __CUDACC__
  for (i=2; i<=n; ++i) for (j=sqrt(i);j>1;--j) if (i%j==0) {--freq; break;}
#else
  for (i=2; i<=n; ++i) for (j=std::sqrt(i);j>1;--j) if (i%j==0) {--freq; break;}
#endif
  return freq;
}

#include <unistd.h>
double GPUTimingTestHelper::launchGPUStreamSleep(unsigned nBlocks, unsigned nThreads, clock64_t nCycles, unsigned milliseconds) {

#ifdef __CUDACC__
	cudaEvent_t start, stop;
	cudaEvent_t startGPU, stopGPU;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);

	cudaStream_t stream;
	cudaStreamCreate( &stream );

	cudaEventRecord(start, 0);
	cudaEventRecord(startGPU,stream);

	kernelGPUSleep<<<nBlocks,nThreads,0,stream>>>(nCycles);
	cudaEventRecord(stopGPU,stream);
	cudaStreamWaitEvent(stream, stopGPU, 0);
#endif

//	// sleep cpu to simulate work

	{
		MonteRay::cpuTimer timer;
		timer.start();
		int f = frequency_of_primes (1999999);
//		printf ("The number of primes lower than 2,000,000 is: %d\n",f);
		timer.stop();
		std::cout << "Debug: gpuTiming_test_helper::launchGPUStreamSleep -- cpu time=" << std::setprecision(5) << timer.getTime() << std::endl;

	}

	float gpuTime;

#ifdef __CUDACC__
	cudaStreamSynchronize( stream );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	gpuErrchk( cudaPeekAtLastError() );
//

	cudaEventElapsedTime(&gpuTime, startGPU, stopGPU );
	std::cout << "Debug: gpuTiming_test_helper::launchGPUStreamSleep -- gpu time=" << gpuTime/1000.0 << std::endl;

	float totalTime;
	cudaEventElapsedTime(&totalTime, start, stop );
	std::cout << "Debug: gpuTiming_test_helper::launchGPUStreamSleep -- total time=" << totalTime/1000.0 << std::endl;
#endif

//	cudaStreamDestroy(stream);
	return gpuTime/1000.0;
}
