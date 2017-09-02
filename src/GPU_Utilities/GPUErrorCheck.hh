#ifndef GPUERRORCHECK_HH_
#define GPUERRORCHECK_HH_

/**
 * \file
 * */

#include <stdio.h>

#include "MonteRayDefinitions.hh"

namespace MonteRay{

#ifdef __CUDACC__

/**
 * \def MonteRayAlwaysPeekAtLastError()
 * \brief Unconditional wrapper to cudaPeekAtLastError()
 */

#define	MonteRayAlwaysPeekAtLastError() {										\
	cudaError_t code = cudaPeekAtLastError();									\
	if( code != cudaSuccess ) {													\
		fprintf(stderr,"GPUassert: Error %s at line %d in file %s\n",  	 		\
				cudaGetErrorString(code), __LINE__, __FILE__);					\
				exit(1);														\
		}																		\
	}

/**
 * \def MonteRaySometimesPeekAtLastError()
 * \brief Preprocessor defined conditional wrapper to cudaPeekAtLastError()
 */

#ifdef NDEBUG
#define	MonteRaySometimesPeekAtLastError() {}
#else
#define	MonteRaySometimesPeekAtLastError() { MonteRayAlwaysPeekAtLastError(); }
#endif

/**
 * \def MonteRayPeekAtLastError(debug)
 * \brief Wrapper to cudaPeekAtLastError()
 *
 * The user can force the call to cudaPeekAtLastError() by passing true
 * to the function.  Otherwise a conditionally compiled MonteRaySometimesPeekAtLastError
 * is called.  MonteRaySometimesPeekAtLastError does nothing if compiled for
 * Release of ReleaseWithDebugInfo.
 *
 * debug should never be set when called after a non-unit test kernel call.
 * This will force a sync between the cpu and the gpu.
 */
#define MONTERAY_PEAKATLASTERROR(value) {                                   \
	bool force_check = value;                                     			\
	if( force_check ) {														\
		MonteRayAlwaysPeekAtLastError();									\
	} else {																\
		MonteRaySometimesPeekAtLastError();									\
	} }


#ifndef NDEBUG
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

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }

#else
#define CUDA_CHECK_RETURN(value) {	value; }
#define gpuErrchk(ans) { ans; }
#endif

#endif

// ABORT

#ifdef __CUDA_ARCH__
__host__ __device__ inline void MonteRayAbort( const char* message, const char *file, int line){
	printf("Error: %s %s %d\n", message, file, line);
	asm("trap;");
}
#else
inline void MonteRayAbort( const char* message, const char *file, int line){
	fprintf(stderr,"Error: %s %s %d\n", message, file, line);
	exit(1);
}
#endif

#define ABORT(message) { MonteRay::MonteRayAbort(message, __FILE__, __LINE__); }

} /* end namespace MonteRay */

#endif /* GPUERRORCHECK_HH_ */
