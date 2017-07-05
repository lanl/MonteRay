#include "gpuTiming_test_helper.hh"

#include <iostream>
#include <ctime>
#include <iomanip>
#include <time.h>

#include "GPUTiming.hh"
#include "MonteRay_timer.hh"

using namespace MonteRay;

GPUTimingTestHelper::GPUTimingTestHelper(){
}

GPUTimingTestHelper::~GPUTimingTestHelper(){
}

__global__ void kernelGPUSleep(clock64_t nCycles, gpuTiming* timer) {
	timer->start = clock64();
	clock64_t cycles = 0;
	clock64_t start = clock();
    while(cycles < nCycles) {
        cycles = clock64() - start;
    }
    timer->stop = clock64();
}

void GPUTimingTestHelper::launchGPUSleep( clock64_t nCycles, MonteRay::gpuTimingHost* timer) {
	cudaEvent_t sync;
	cudaEventCreate(&sync);
	kernelGPUSleep<<<1,1>>>(nCycles, timer->ptr_device);
	gpuErrchk( cudaPeekAtLastError() );
	cudaEventRecord(sync, 0);
	cudaEventSynchronize(sync);

	return;
}


__global__ void kernelGPUSleep(clock64_t nCycles) {
	clock64_t cycles = 0;
	clock64_t start = clock64();
    while(cycles < nCycles) {
        cycles = clock64() - start;
    }
}

int frequency_of_primes (int n) {
  int i,j;
  int freq=n-1;
  for (i=2; i<=n; ++i) for (j=sqrt(i);j>1;--j) if (i%j==0) {--freq; break;}
  return freq;
}

#include <unistd.h>
double GPUTimingTestHelper::launchGPUStreamSleep(unsigned nBlocks, unsigned nThreads, clock64_t nCycles, unsigned milliseconds) {

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

//	// sleep cpu to simulate work

	{
		MonteRay::cpuTimer timer;
		timer.start();
		int f = frequency_of_primes (1999999);
//		printf ("The number of primes lower than 2,000,000 is: %d\n",f);
		timer.stop();
		std::cout << "Debug: gpuTiming_test_helper::launchGPUStreamSleep -- cpu time=" << std::setprecision(5) << timer.getTime() << std::endl;

	}

	cudaStreamSynchronize( stream );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	gpuErrchk( cudaPeekAtLastError() );
//
	float gpuTime;
	cudaEventElapsedTime(&gpuTime, startGPU, stopGPU );
	std::cout << "Debug: gpuTiming_test_helper::launchGPUStreamSleep -- gpu time=" << gpuTime/1000.0 << std::endl;

	float totalTime;
	cudaEventElapsedTime(&totalTime, start, stop );
	std::cout << "Debug: gpuTiming_test_helper::launchGPUStreamSleep -- total time=" << totalTime/1000.0 << std::endl;

//	cudaStreamDestroy(stream);
	return gpuTime/1000.0;
}
