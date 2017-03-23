/*
 * GPUSync.hh
 *
 *  Created on: Mar 20, 2017
 *      Author: jsweezy
 */

#ifndef GPUSYNC_HH_
#define GPUSYNC_HH_

#ifdef CUDA
#include <cuda.h>
#endif

#include "MonteRayDefinitions.hh"

namespace MonteRay {

class GPUSync {
public:
	GPUSync(){
#ifdef CUDA
		cudaEventCreate(&sync_event);
#endif
	}

	~GPUSync(){
#ifdef CUDA
		cudaEventDestroy(sync_event);
#endif
	}

	void sync(){
#ifdef CUDA
		cudaEventRecord(sync_event, 0);
		cudaEventSynchronize(sync_event);
#endif
	}

private:
	cudaEvent_t sync_event;

};

} /* namespace MonteRay */

#endif /* GPUSYNC_HH_ */
