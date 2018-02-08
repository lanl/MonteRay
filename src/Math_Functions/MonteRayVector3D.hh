#ifndef MONTERAYVECTOR3D_HH_
#define MONTERAYVECTOR3D_HH_

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

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
    CUDA_CALLABLE_MEMBER
    Vector3D( void ) {
        elems[ 0 ] = T();
        elems[ 1 ] = T();
        elems[ 2 ] = T();
    }

    CUDA_CALLABLE_MEMBER
	Vector3D( T* init_elems) {
	    elems[ 0 ] = init_elems[ 0 ];
        elems[ 1 ] = init_elems[ 1 ];
        elems[ 2 ] = init_elems[ 2 ];
	}

    CUDA_CALLABLE_MEMBER
	Vector3D( const Vector3D& rhs ) {
        elems[ 0 ] = rhs.elems[ 0 ];
        elems[ 1 ] = rhs.elems[ 1 ];
        elems[ 2 ] = rhs.elems[ 2 ];
	}

    CUDA_CALLABLE_MEMBER
    Vector3D& operator=( const Vector3D& rhs ) {
        elems[ 0 ] = rhs.elems[ 0 ];
        elems[ 1 ] = rhs.elems[ 1 ];
        elems[ 2 ] = rhs.elems[ 2 ];
        return *this;
    }

    CUDA_CALLABLE_MEMBER
	Vector3D( T value ) {
	    elems[ 0 ] = elems[ 1 ] = elems[ 2 ] = value;
	}

    CUDA_CALLABLE_MEMBER
	Vector3D( T t0, T t1, T t2 ) {
		elems[ 0 ] = t0;
		elems[ 1 ] = t1;
		elems[ 2 ] = t2;
	}

    CUDA_CALLABLE_MEMBER
	~Vector3D() {}

    CUDA_CALLABLE_MEMBER
	T operator[]( unsigned index ) const {
		if( index < 3 ) {
			return elems[ index ];
		} else {
			return T();
		}
	}

    CUDA_CALLABLE_MEMBER
    T& operator[]( unsigned index ) {
        if( index < 3 ) {
            return elems[ index ];
        } else {
#ifndef __CUDA_ARCH__
        	throw std::runtime_error("Vector3D::operator[] has thrown an exception. index = > 3");
#else
        	return elems[ 0 ]; // silence the voices
#endif
        }

    }

    CUDA_CALLABLE_MEMBER
    const T* data( void ) const { return elems; }

    CUDA_CALLABLE_MEMBER
	Vector3D& normalize( void ) {
	    T mag = magnitude();
	    if( mag > 0.0 ) {
	      return (*this) /= magnitude();
	    } else {
	        return *this;
//	    	throw std::runtime_error("Vector3D::normalize() has thrown an exception. Magnitude is 0.0");
	    }
	}

    CUDA_CALLABLE_MEMBER
    Vector3D normal( void ) const {
        T mag = magnitude();
        if( mag > 0.0 ) {
            Vector3D<T> temp = (*this) * (1/mag);
          return temp;
        } else {
            return *this;
//            throw std::runtime_error("Vector3D::normal() has thrown an exception. Magnitude is 0.0");
        }
    }

    CUDA_CALLABLE_MEMBER
	T magnitude( void ) const {
	    return std::sqrt( elems[ 0 ] * elems[ 0 ] + elems[ 1 ] * elems[ 1 ] + elems[ 2 ]*elems[ 2 ] );
	}

    CUDA_CALLABLE_MEMBER
	Vector3D& operator/=( T value ) { (*this) *= 1/value; return *this; }

    CUDA_CALLABLE_MEMBER
    Vector3D& operator+=( T value ) {
        elems[ 0 ] += value;
        elems[ 1 ] += value;
        elems[ 2 ] += value;
        return *this;
    }

    CUDA_CALLABLE_MEMBER
    Vector3D& operator*=( T value ) {
        elems[ 0 ] *= value;
        elems[ 1 ] *= value;
        elems[ 2 ] *= value;
        return *this;
    }

    CUDA_CALLABLE_MEMBER
    Vector3D operator*( T value ) const {
        Vector3D<T> tmp(*this);
        tmp.elems[ 0 ] *= value;
        tmp.elems[ 1 ] *= value;
        tmp.elems[ 2 ] *= value;
        return std::move(tmp);
    }


    template< typename T1 >
    CUDA_CALLABLE_MEMBER
    Vector3D& operator+=( const Vector3D<T1>& rhs) {
        for( unsigned d=0; d<3; ++d )
            elems[d] += rhs[d];
        return *this;
    }

    CUDA_CALLABLE_MEMBER
    Vector3D& operator+=( const Vector3D& rhs ) {
        elems[ 0 ] += rhs.elems[ 0 ];
        elems[ 1 ] += rhs.elems[ 1 ];
        elems[ 2 ] += rhs.elems[ 2 ];
        return *this;
    }

    CUDA_CALLABLE_MEMBER
    Vector3D& operator-=( const Vector3D& rhs ) {
        elems[ 0 ] -= rhs.elems[ 0 ];
        elems[ 1 ] -= rhs.elems[ 1 ];
        elems[ 2 ] -= rhs.elems[ 2 ];
        return *this;
    }

    CUDA_CALLABLE_MEMBER
    Vector3D& operator-=( T Value ) { (*this) += -Value; return *this; }

    CUDA_CALLABLE_MEMBER
    Vector3D operator+( const Vector3D& that ) const {
    	Vector3D<T> temp( *this );
    	temp += that;
    	return std::move(temp);
    }

    CUDA_CALLABLE_MEMBER
    Vector3D operator-( const Vector3D& that ) const {
    	Vector3D<T> temp( *this );
    	temp -= that;
    	return std::move(temp);
    }

    CUDA_CALLABLE_MEMBER
    T dot( const Vector3D& rhs ) const {
        return elems[ 0 ] * rhs.elems[ 0 ] + elems[ 1 ] * rhs.elems[ 1 ] + elems[ 2 ] * rhs.elems[ 2 ];
    }

    CUDA_CALLABLE_MEMBER
    Vector3D cross( const Vector3D& rhs ) const {
    	Vector3D<T> temp;
    	temp[0] = elems[1]* rhs.elems[2] - elems[2]* rhs.elems[1];
    	temp[1] = elems[2]* rhs.elems[0] - elems[0]* rhs.elems[2];
    	temp[2] = elems[0]* rhs.elems[1] - elems[1]* rhs.elems[0];
    	return std::move(temp);
    }

};

template< typename T>
CUDA_CALLABLE_MEMBER
Vector3D<T> operator*( const T& lhs, const Vector3D<T>& rhs)
{ return rhs*lhs;}

template< typename T>
CUDA_CALLABLE_MEMBER
T dot( const Vector3D<T>& vec1, const Vector3D<T>& vec2 )
{ return vec1.dot( vec2 ); }

template< typename T>
CUDA_CALLABLE_MEMBER
Vector3D<T> cross( const Vector3D<T>& vec1, const Vector3D<T>& vec2 )
{ return vec1.cross( vec2 ); }

template< typename T>
CUDA_CALLABLE_MEMBER
bool operator==( const Vector3D<T>& rhs, const Vector3D<T>& lhs) {
    return (rhs[0]==lhs[0]) && (rhs[1]==lhs[1]) && (rhs[2] == lhs[2]);
}

template< typename T>
CUDA_CALLABLE_MEMBER
Vector3D<T> operator/( const Vector3D<T>& v1, const Vector3D<T>& v2) {
    return Vector3D<T>( v1 ) /= v2;
}

template< typename T>
CUDA_CALLABLE_MEMBER
Vector3D<T> operator/( const Vector3D<T>& v1, double factor ) {
    return Vector3D<T>( v1 ) /= factor;
}

template< typename T>
CUDA_CALLABLE_MEMBER
bool operator!=( const Vector3D<T>& rhs, const Vector3D<T>& lhs) { return !(rhs==lhs); }

template< typename T>
CUDA_CALLABLE_MEMBER
std::ostream& operator << ( std::ostream& os, const Vector3D<T>& arg){
    os << "( " << arg[0] << ", "<< arg[1] << ", "<< arg[2]<<" )";
    return os;
}
}
#endif /*VECTOR3D_HH_*/
