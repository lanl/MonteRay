#ifndef VECTOR3D_H_
#define VECTOR3D_H_

#include "global.h"
#include <math.h>

namespace MonteRay{

class Vector3D {
public:
	typedef global::float_t float_t;
	typedef float_t T;

	T elems[ 3 ];

	Vector3D( void ) {
		elems[ 0 ] = T();
		elems[ 1 ] = T();
		elems[ 2 ] = T();
	}

	Vector3D( T* init_elems) {
		elems[ 0 ] = init_elems[ 0 ];
		elems[ 1 ] = init_elems[ 1 ];
		elems[ 2 ] = init_elems[ 2 ];
	}

	Vector3D( const Vector3D& rhs ) {
		elems[ 0 ] = rhs.elems[ 0 ];
		elems[ 1 ] = rhs.elems[ 1 ];
		elems[ 2 ] = rhs.elems[ 2 ];
	}

	Vector3D& operator=( const Vector3D& rhs ) {
		elems[ 0 ] = rhs.elems[ 0 ];
		elems[ 1 ] = rhs.elems[ 1 ];
		elems[ 2 ] = rhs.elems[ 2 ];
		return *this;
	}

	Vector3D( T value ) {
		elems[ 0 ] = elems[ 1 ] = elems[ 2 ] = value;
	}

	Vector3D( T t0, T t1, T t2 ) {
		elems[ 0 ] = t0;
		elems[ 1 ] = t1;
		elems[ 2 ] = t2;
	}

	~Vector3D() {}

	T operator[]( unsigned index ) const {
		return elems[ index ];
	}

	T& operator[]( unsigned index ) {
		return elems[ index ];
	}

	const T* data( void ) const { return elems; }

	Vector3D& normalize( void ) {
		T mag = magnitude();
		return (*this) /= magnitude();
	}

	Vector3D normal( void ) const {
		T mag = magnitude();
		Vector3D temp = (*this) * (1/mag);
		return temp;
	}

	T magnitude( void ) const {
		return sqrt( elems[ 0 ] * elems[ 0 ] + elems[ 1 ] * elems[ 1 ] + elems[ 2 ]*elems[ 2 ] );
	}

	Vector3D& operator/=( T value ) { (*this) *= 1/value; return *this; }

	Vector3D& operator+=( T value ) {
		elems[ 0 ] += value;
		elems[ 1 ] += value;
		elems[ 2 ] += value;
		return *this;
	}
	Vector3D& operator*=( T value ) {
		elems[ 0 ] *= value;
		elems[ 1 ] *= value;
		elems[ 2 ] *= value;
		return *this;
	}

	Vector3D operator*( T value ) const {
		Vector3D tmp(*this);
		tmp.elems[ 0 ] *= value;
		tmp.elems[ 1 ] *= value;
		tmp.elems[ 2 ] *= value;
		return tmp;
	}

	Vector3D& operator+=( const Vector3D& rhs ) {
		elems[ 0 ] += rhs.elems[ 0 ];
		elems[ 1 ] += rhs.elems[ 1 ];
		elems[ 2 ] += rhs.elems[ 2 ];
		return *this;
	}

	Vector3D& operator-=( const Vector3D& rhs ) {
		elems[ 0 ] -= rhs.elems[ 0 ];
		elems[ 1 ] -= rhs.elems[ 1 ];
		elems[ 2 ] -= rhs.elems[ 2 ];
		return *this;
	}

	Vector3D& operator-=( T Value ) { (*this) += -Value; return *this; }

	Vector3D operator+( const Vector3D& that ) const {
		Vector3D temp( *this );
		temp += that;
		return temp;
	}

	Vector3D operator-( const Vector3D& that ) const {
		Vector3D temp( *this );
		temp -= that;
		return temp;
	}


	T dot( const Vector3D& rhs ) const {
		return elems[ 0 ] * rhs.elems[ 0 ] + elems[ 1 ] * rhs.elems[ 1 ] + elems[ 2 ] * rhs.elems[ 2 ];
	}

	Vector3D cross( const Vector3D& rhs ) const {
		Vector3D temp;
		temp[0] = elems[1]* rhs.elems[2] - elems[2]* rhs.elems[1];
		temp[1] = elems[2]* rhs.elems[0] - elems[0]* rhs.elems[2];
		temp[2] = elems[0]* rhs.elems[1] - elems[1]* rhs.elems[0];
		return temp;
	}

private:

};

typedef Vector3D Position_t;
typedef Vector3D Direction_t;

//Vector3D operator*( const global::float_t& lhs, const Vector3D& rhs)
//{ return rhs*lhs;}
//
//global::float_t dot( const Vector3D& vec1, const Vector3D& vec2 )
//{ return vec1.dot( vec2 ); }
//
//Vector3D cross( const Vector3D& vec1, const Vector3D& vec2 )
//{ return vec1.cross( vec2 ); }
//
//bool operator==( const Vector3D& rhs, const Vector3D& lhs) {
//	return (rhs[0]==lhs[0]) && (rhs[1]==lhs[1]) && (rhs[2] == lhs[2]);
//}
//
////Vector3D operator/( const Vector3D& v1, const Vector3D& v2) {
////	return Vector3D( v1 ) /= v2;
////}
//
//Vector3D operator/( const Vector3D& v1, double factor ) {
//	return Vector3D( v1 ) /= factor;
//}
//
//bool operator!=( const Vector3D& rhs, const Vector3D& lhs) { return !(rhs==lhs); }
//

}

#endif /* VECTOR3D_H_ */
