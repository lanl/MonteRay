#include <cstring>

#include "MonteRayDefinitions.hh"

#include "gpuDistanceCalculator_test_helper.hh"

namespace MonteRay{


gpuDistanceCalculatorTestHelper::gpuDistanceCalculatorTestHelper(){
    pRayInfo.reset( new RayWorkInfo( 1, true ) );
}

void gpuDistanceCalculatorTestHelper::gpuCheck() {
    //MonteRay::gpuCheck();
}

gpuDistanceCalculatorTestHelper::~gpuDistanceCalculatorTestHelper(){

    //	std::cout << "Debug: starting ~gpuDistanceCalculatorTestHelper()" << std::endl;

}

unsigned gpuDistanceCalculatorTestHelper:: getNumCrossingsFromGPU(void) {
    return pRayInfo->getRayCastSize(0);
}


void gpuDistanceCalculatorTestHelper::setupTimers(){
#ifdef __CUDACC__
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#else
    timer.start();
#endif
}

void gpuDistanceCalculatorTestHelper::stopTimers(){
#ifdef __CUDACC__
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;

    cudaEventElapsedTime(&elapsedTime, start, stop );
    std::cout << "Elapsed time in CUDA kernel=" << elapsedTime << " msec" << std::endl;
#else
    timer.stop();
    std::cout << "Elapsed time in non-CUDA kernel=" << timer.getTime()*1000.0  << " msec" << std::endl;
#endif


}

}
