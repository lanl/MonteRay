#ifndef FASTERHASH_HH_
#define FASTERHASH_HH_

#include "MonteRayTypes.hh"
#include "MonteRayConstants.hh"
#include "MonteRayAssert.hh"


namespace MonteRay {

/// \brief A very fast hash that calculates log2 spaced integer bin numbers using bit shifting
class FasterHash {

private:
    // The 8 significand bits provide 5964 bins over the range 1e-11 to 1e3
    static constexpr CUDA_CALLABLE_MEMBER unsigned DESIRED_SIGNIFICANDS(){ return 8; }

    static constexpr CUDA_CALLABLE_MEMBER unsigned DOUBLE_EXPONENT_BITS(){ return 11; }
    static constexpr CUDA_CALLABLE_MEMBER unsigned FLOAT_EXPONENT_BITS(){  return 8; }
    static constexpr CUDA_CALLABLE_MEMBER unsigned DOUBLE_SHIFT(){ return 64-(DESIRED_SIGNIFICANDS() + DOUBLE_EXPONENT_BITS()); }
    static constexpr CUDA_CALLABLE_MEMBER unsigned FLOAT_SHIFT(){ return 32-(DESIRED_SIGNIFICANDS() + FLOAT_EXPONENT_BITS()); }

public:

    CUDA_CALLABLE_MEMBER
    static size_t hashFunction(const double value) {
        // For double
        // shifts the bits and returns binary equivalent integer
        //std::cout << "Debug -- Calling hashFunction(double)\n";

        MONTERAY_ASSERT_MSG( value >= 0.0, "Negative values are not allowed.");

        std::uint64_t i;
        std::memcpy(&i, &value, sizeof(value));
        return i >> DOUBLE_SHIFT();
    }

    CUDA_CALLABLE_MEMBER
    static size_t hashFunction(const float value) {
        // For float
        // shifts the bits and returns binary equivalent integer
        //std::cout << "Debug -- Calling hashFunction(float)\n";

        MONTERAY_ASSERT_MSG( value >= 0.0, "Negative values are not allowed.");

        std::uint32_t i;
        std::memcpy(&i, &value, sizeof(value));
        return i >> FLOAT_SHIFT();
    }

    template < typename TV, typename std::enable_if< sizeof(TV) == 8 >::type* = nullptr >
    CUDA_CALLABLE_MEMBER
    static
    TV invHashFunction( const size_t index, const size_t minIndex = 0 ) {
        //std::cout << "Debug -- Calling invHashFunction(index)->double\n";

        uint64_t i = index + minIndex << DOUBLE_SHIFT();
        TV value;
        memcpy(&value,&i, sizeof(i) );

#ifndef __CUDA_ARCH__
        MONTERAY_ASSERT_MSG(std::isnormal(value), "Value at hash bin is not a normal value." );
#endif
        return value;
    }

    template < typename TV, typename std::enable_if< sizeof(TV) == 4 >::type* = nullptr >
    CUDA_CALLABLE_MEMBER
    static
    TV invHashFunction( const size_t index, const size_t minIndex = 0 ) {
        //std::cout << "Debug -- Calling invHashFunction(index)->float\n";

        uint32_t i = index + minIndex << FLOAT_SHIFT();
        TV value;
        memcpy(&value,&i, sizeof(i) );

#ifndef __CUDA_ARCH__
        MONTERAY_ASSERT_MSG(std::isnormal(value), "Value at hash bin is not a normal value." );
#endif
        return value;
    }

};

} // end namespace


#endif /* FASTERHASH_HH_ */
