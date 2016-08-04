#ifndef GPUGLOBAL_HH_
#define GPUGLOBAL_HH_

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#ifdef CUDA
#include <cuda.h>
#endif
#include "/projects/opt/centos7/cuda/7.5/include/driver_types.h"

namespace MonteRay{
typedef float gpuFloatType_t;
typedef float gpuTallyType_t;

const gpuFloatType_t gpu_neutron_molar_mass = 1.00866491597f;
const gpuFloatType_t gpu_AvogadroBarn = .602214179f;
typedef long long clock64_t;

void cudaReset(void);

void gpuCheck();

#ifdef CUDA
// From: https://www.sharcnet.ca/help/index.php/CUDA_tips_and_tricks
__device__ inline void atomicAddDouble (double *address, double value){
  unsigned long long oldval, newval, readback;

  oldval = __double_as_longlong(*address);
  newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
    {
     oldval = readback;
     newval = __double_as_longlong(__longlong_as_double(oldval) + value);
    }
}
#endif

void setCudaPrintBufferSize( size_t size);


}

#ifdef CUDA
/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

namespace MonteRay{
class gpuSync {
public:
	gpuSync();
	~gpuSync();

	void sync();
private:
private:
	cudaEvent_t sync_event;
};
}

#endif /* GPUGLOBAL_HH_ */
