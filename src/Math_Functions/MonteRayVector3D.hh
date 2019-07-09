#ifndef MONTERAYVECTOR3D_HH_
#define MONTERAYVECTOR3D_HH_

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <tuple>

#include "MonteRayTypes.hh"
#include "MonteRayAssert.hh"

namespace MonteRay{

/// Class Vector3D (From MCATK)
///
/// Defines the mathematics for a 3 element vector.\n
///  Scalar * vector = vector\n
///  vector * scalar = vector\n
///  Magnitude, normalize, and dot are all defined.\n
///  Math can be implemented as needed.\n
template< typename T >
class Vector3D
{
    T elems[ 3 ];
public:
    CUDA_CALLABLE_MEMBER constexpr Vector3D( void ) {
        elems[ 0 ] = T();
        elems[ 1 ] = T();
        elems[ 2 ] = T();
    }

    CUDA_CALLABLE_MEMBER constexpr Vector3D( T* init_elems) {
        elems[ 0 ] = init_elems[ 0 ];
        elems[ 1 ] = init_elems[ 1 ];
        elems[ 2 ] = init_elems[ 2 ];
    }

    CUDA_CALLABLE_MEMBER constexpr Vector3D( T value ) {
        elems[ 0 ] = elems[ 1 ] = elems[ 2 ] = value;
    }

    CUDA_CALLABLE_MEMBER constexpr Vector3D( T t0, T t1, T t2 ) {
        elems[ 0 ] = t0;
        elems[ 1 ] = t1;
        elems[ 2 ] = t2;
    }
constexpr const T& operator[]( unsigned index ) const { MONTERAY_ASSERT(index < 3); return elems[ index ]; }
    constexpr T& operator[]( unsigned index ) { MONTERAY_ASSERT(index < 3); return elems[ index ]; }

    constexpr const T* data( void ) const { return elems; }

    constexpr Vector3D& normalize( void ) {
      T mag = magnitude();
      MONTERAY_ASSERT(mag > 0.0);
      return (*this) /= magnitude();
    }

    constexpr Vector3D normal( void ) const {
      T mag = magnitude();
      MONTERAY_ASSERT(mag > 0.0);
      Vector3D<T> temp = (*this) * (static_cast<T>(1.0)/mag);
      return temp;
    }

    constexpr T magnitude( void ) const {
        return std::sqrt( elems[ 0 ] * elems[ 0 ] + elems[ 1 ] * elems[ 1 ] + elems[ 2 ]*elems[ 2 ] );
    }

    constexpr Vector3D& operator/=( T value ) { (*this) *= 1/value; return *this; }

    constexpr Vector3D& operator+=( T value ) {
        elems[ 0 ] += value;
        elems[ 1 ] += value;
        elems[ 2 ] += value;
        return *this;
    }

    constexpr Vector3D& operator*=( T value ) {
        elems[ 0 ] *= value;
        elems[ 1 ] *= value;
        elems[ 2 ] *= value;
        return *this;
    }

    constexpr Vector3D operator*( T value ) const {
        Vector3D<T> tmp(*this);
        tmp.elems[ 0 ] *= value;
        tmp.elems[ 1 ] *= value;
        tmp.elems[ 2 ] *= value;
        return std::move(tmp);
    }


    template< typename T1 >
    constexpr Vector3D& operator+=( const Vector3D<T1>& rhs) {
        for( unsigned d=0; d<3; ++d )
            elems[d] += rhs[d];
        return *this;
    }

    constexpr Vector3D& operator+=( const Vector3D& rhs ) {
        elems[ 0 ] += rhs.elems[ 0 ];
        elems[ 1 ] += rhs.elems[ 1 ];
        elems[ 2 ] += rhs.elems[ 2 ];
        return *this;
    }

    constexpr Vector3D& operator-=( const Vector3D& rhs ) {
        elems[ 0 ] -= rhs.elems[ 0 ];
        elems[ 1 ] -= rhs.elems[ 1 ];
        elems[ 2 ] -= rhs.elems[ 2 ];
        return *this;
    }

    constexpr Vector3D& operator-=( T Value ) { (*this) += -Value; return *this; }

    constexpr Vector3D operator+( const Vector3D& that ) const {
        Vector3D<T> temp( *this );
        temp += that;
        return std::move(temp);
    }

    constexpr Vector3D operator-( const Vector3D& that ) const {
        Vector3D<T> temp( *this );
        temp -= that;
        return std::move(temp);
    }

    constexpr T dot( const Vector3D& rhs ) const {
        return elems[ 0 ] * rhs.elems[ 0 ] + elems[ 1 ] * rhs.elems[ 1 ] + elems[ 2 ] * rhs.elems[ 2 ];
    }

    constexpr Vector3D cross( const Vector3D& rhs ) const {
        Vector3D<T> temp;
        temp[0] = elems[1]* rhs.elems[2] - elems[2]* rhs.elems[1];
        temp[1] = elems[2]* rhs.elems[0] - elems[0]* rhs.elems[2];
        temp[2] = elems[0]* rhs.elems[1] - elems[1]* rhs.elems[0];
        return std::move(temp);
    }

};

template< typename T>
constexpr Vector3D<T> operator*( const T& lhs, const Vector3D<T>& rhs)
{ return rhs*lhs;}

template< typename T>
constexpr T dot( const Vector3D<T>& vec1, const Vector3D<T>& vec2 )
{ return vec1.dot( vec2 ); }

template< typename T>
constexpr Vector3D<T> cross( const Vector3D<T>& vec1, const Vector3D<T>& vec2 )
{ return vec1.cross( vec2 ); }

template< typename T>
constexpr bool operator==( const Vector3D<T>& rhs, const Vector3D<T>& lhs) {
    return (rhs[0]==lhs[0]) && (rhs[1]==lhs[1]) && (rhs[2] == lhs[2]);
}

template< typename T>
constexpr Vector3D<T> operator/( const Vector3D<T>& v1, const Vector3D<T>& v2) {
    return Vector3D<T>( v1 ) /= v2;
}

template< typename T>
constexpr Vector3D<T> operator/( const Vector3D<T>& v1, double factor ) {
    return Vector3D<T>( v1 ) /= factor;
}

template< typename T>
constexpr bool operator!=( const Vector3D<T>& rhs, const Vector3D<T>& lhs) { return !(rhs==lhs); }

template< typename T>
constexpr std::ostream& operator << ( std::ostream& os, const Vector3D<T>& arg){
    os << "( " << arg[0] << ", "<< arg[1] << ", "<< arg[2]<<" )";
    return os;
}

template <typename T>
constexpr auto getDistanceDirection(
        const MonteRay::Vector3D<T>& pos,
        const MonteRay::Vector3D<T>& pos2) {
    auto dir = pos2 - pos;
    auto dist = dir.magnitude();
    auto invDistance = static_cast<decltype(dist)>(1.0)/dist;
    dir *= invDistance;
    return std::make_tuple(dist, dir);
}

} /* namespace MonteRay */
#endif /*VECTOR3D_HH_*/
