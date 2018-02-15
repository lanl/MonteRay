/*
 * GPUUtilityFunctions.hh
 *
 *  Created on: Mar 20, 2017
 *      Author: jsweezy
 */

#ifndef GPUUTILITYFUNCTIONS_HH_
#define GPUUTILITYFUNCTIONS_HH_

#include <cstddef>

namespace MonteRay {

void cudaReset(void);
void gpuReset();
void gpuCheck();
void gpuInfo();
int getNumberOfGPUS(void);
void setCudaPrintBufferSize( size_t size);

} /* namespace MonteRay */

#endif /* GPUUTILITYFUNCTIONS_HH_ */
