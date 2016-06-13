#ifndef GPUGLOBAL_HH_
#define GPUGLOBAL_HH_

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#ifdef CUDA
#include <cuda.h>
#endif

typedef float gpuFloatType_t;
const gpuFloatType_t gpu_neutron_molar_mass = 1.00866491597f;
const gpuFloatType_t gpu_AvogadroBarn = .602214179f;


void cudaReset(void);

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

#endif /* GPUGLOBAL_HH_ */
