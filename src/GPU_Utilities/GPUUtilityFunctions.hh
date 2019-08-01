/*
 * GPUUtilityFunctions.hh
 *
 *  Created on: Mar 20, 2017
 *      Author: jsweezy
 */

#ifndef GPUUTILITYFUNCTIONS_HH_
#define GPUUTILITYFUNCTIONS_HH_

#include <cstddef>
#include <utility>
#include "Invokers.hh"

namespace MonteRay {

void cudaReset(const bool verbose = false);
void gpuReset(const bool verbose = false);
void gpuCheck( const bool verbose );
void gpuCheck();

void gpuInfo();
int getNumberOfGPUS(const bool verbose = false);
void setCudaDevice(int deviceID, const bool verbose = false);
int getCudaDevice( const bool verbose );
void setCudaPrintBufferSize( size_t size, const bool verbose = false);
void setCudaStackSize( size_t size, const bool verbose = false );

void defaultStreamSync();
void deviceSynchronize();

std::pair<unsigned, unsigned> setLaunchBounds( int nThreads, int nRaysPerThread, const unsigned numRays );

} /* namespace MonteRay */

#endif /* GPUUTILITYFUNCTIONS_HH_ */
