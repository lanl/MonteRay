#include "MonteRayDefinitions.hh"
namespace MonteRay{
template <typename Func>
CUDA_CALLABLE_KERNEL d_invoker(Func f, int i) { f(i); }

template <typename Func>
CUDA_CALLABLE_KERNEL d_invoker(Func f) { f(); }
} // end namespace MonteRay
