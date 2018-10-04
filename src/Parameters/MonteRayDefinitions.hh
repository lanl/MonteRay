#ifndef MONTERAYDEFINITIONS_HH_
#define MONTERAYDEFINITIONS_HH_

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#include <cuda.h>
#else
//typedef unsigned cudaStream_t;
#endif

#include "MonteRayPreprocessorDefinitions.hh"
#include "MonteRayTypes.hh"

#endif /* MONTERAYDEFINITIONS_HH_ */
