#include "MonteRayDefinitions.hh"
namespace MonteRay{
template <typename Func, typename... Args>
CUDA_CALLABLE_KERNEL d_invoker(Func f, Args... args) {
  f(args...); 
}
} // end namespace MonteRay
