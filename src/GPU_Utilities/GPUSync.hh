#ifndef GPUSYNC_HH_
#define GPUSYNC_HH_

#include "MonteRayDefinitions.hh"

namespace MonteRay {

class GPUSync {
public:
	GPUSync(){
#ifdef __CUDACC__
		cudaEventCreate(&sync_event);
#endif
	}

	~GPUSync(){
#ifdef __CUDACC__
		cudaEventDestroy(sync_event);
#endif
	}

	void sync(){
#ifdef __CUDACC__
		cudaEventRecord(sync_event, 0);
		cudaEventSynchronize(sync_event);
#endif
	}

private:
#ifdef __CUDACC__
	cudaEvent_t sync_event;
#endif

};

} /* namespace MonteRay */

#endif /* GPUSYNC_HH_ */
