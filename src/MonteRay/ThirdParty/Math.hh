#ifndef _CUDA_MATH_HPP_
#define _CUDA_MATH_HPP_
// Issue is NVIDA made different functions for floats rather than overload a common function name, much like the old math.h lib.
// This creates the proper overloads if we're inside CUDA code (ifdef __CUDA_ARCH__)
// Also, nvidia populates the global namespace with its math functions rather than putting them in a namespace.
//
#include <cmath>
namespace Math{
#ifdef __CUDA_ARCH__
  constexpr float   exp(float val) { return expf(val); }
  constexpr double  exp(double val) { return ::exp(val); }
  constexpr float   log(float val) { return logf(val); }
  constexpr double  log(double val) { return ::log(val); }
  constexpr float   sqrt(float val) { return sqrtf(val); }
  constexpr double  sqrt(double val) { return ::sqrt(val); }
  constexpr float   cos(float val) {return cosf(val);}
  constexpr double  cos(double val) {return ::cos(val);}
  constexpr float   sin(float val) {return sinf(val);}
  constexpr double  sin(double val) {return ::sin(val);}

  constexpr float   fabs(float val) {return fabsf(val);}
  constexpr double  fabs(double val) {return ::fabs(val);}
  template <typename T>
  constexpr T abs(T val) {return ::abs(val);}
  constexpr float   abs(float val) {return fabsf(val);}
  constexpr double  abs(double val) {return ::fabs(val);}

  constexpr float   floor(float val) {return floorf(val);}
  constexpr double  floor(double val) {return ::floor(val);}
  constexpr float   ceil(float val) {return ceilf(val);}
  constexpr double  ceil(double val) {return ::ceil(val);}

  template <typename T>
  constexpr T min(T val) {return ::min(val);}
  constexpr float   fmin(float val1, float val2) {return fminf(val1, val2);}
  constexpr double  fmin(double val1, double val2) {return ::fmin(val1, val2);}
  constexpr float   min(float val1, float val2) {return fminf(val1, val2);}
  constexpr double  min(double val1, double val2) {return ::fmin(val1, val2);}

  template <typename T>
  constexpr T max(T val) {return ::max(val);}
  constexpr float   fmax(float val1, float val2) {return fmaxf(val1, val2);}
  constexpr double  fmax(double val1, double val2) {return ::fmax(val1, val2);}
  constexpr float   max(float val1, float val2) {return fmaxf(val1, val2);}
  constexpr double  max(double val1, double val2) {return ::fmax(val1, val2);}

  template <typename T>
  constexpr auto signbit(T val) {return ::signbit(val);}
  constexpr auto signbit(float val) {return ::signbit(val);} // explicitly defined to avoid ICE on CUDA 10.1
  constexpr auto signbit(double val) {return ::signbit(val);}

  template <typename T>
  constexpr auto copysign(T&& val) {return ::copysignf(std::forward<T>(val));}

#else
  using std::exp;
  using std::sqrt;
  using std::log;
  using std::cos;
  using std::sin;
  using std::fabs;
  using std::abs;
  using std::floor;
  using std::fmin;
  using std::fmax;
  using std::min;
  using std::max;
  using std::signbit;
  using std::copysign;
#endif
} // end namespace Math
#endif
