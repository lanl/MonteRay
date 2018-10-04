#ifndef MONTERAYASSERT_HH_
#define MONTERAYASSERT_HH_

#include <sstream>

#include "MonteRayTypes.hh"
#include <stdexcept>

namespace MonteRay {
// ABORT
#ifdef __CUDA_ARCH__
__device__ inline void MonteRayAbort( const char* message, const char *file, int line){
    printf("Error: %s %s %d\n", message, file, line);
    asm("trap;");
}
#else
CUDAHOST_CALLABLE_MEMBER inline void MonteRayAbort( const char* message, const char *file, int line){
    std::stringstream msg;
    msg << message << "\n";
    msg << "Called from : " << file << "[" << line << "] \n\n";
    throw std::runtime_error( msg.str() );
}
#endif

#define ABORT(message) { MonteRay::MonteRayAbort(message, __FILE__, __LINE__); }

CUDA_CALLABLE_MEMBER inline void MonteRayAssert( bool test, const char *file, int line){
    if( test != true ) {
        MonteRayAbort( "MonteRay ERROR:", file, line );
    }
}

#ifndef NDEBUG
#define MONTERAY_ASSERT(test) { MonteRayAssert(test, __FILE__, __LINE__); }
#else
#define MONTERAY_ASSERT(test) { (void)0; }
#endif

CUDA_CALLABLE_MEMBER inline void MonteRayAssertMsg( bool test, const char* message, const char *file, int line){
    if( test != true ) {
        MonteRayAbort( message, file, line );
    }
}

#ifndef NDEBUG
#define MONTERAY_ASSERT_MSG(test, message) { MonteRayAssertMsg(test, message, __FILE__, __LINE__); }
#else
#define MONTERAY_ASSERT_MSG(test, message) { (void)0; }
#endif

CUDA_CALLABLE_MEMBER inline void MonteRayVerify( bool test, const char* message, const char *file, int line){
    if( test != true ) {
        MonteRayAbort( message, file, line );
    }
}

#define MONTERAY_VERIFY(test, message) { MonteRay::MonteRayVerify(test, message, __FILE__, __LINE__); }

} // end namespace MonteRay

#endif /* MONTERAYASSERT_HH_ */
