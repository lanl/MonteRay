#ifndef MONTERAY_MATERIALSPEC_HH_
#define MONTERAY_MATERIALSPEC_HH_

#include <limits>

#include "MonteRayDefinitions.hh"

namespace MonteRay {

///\brief Material Spec is a structure that holds definition of a Material ID and density.
class MonteRay_MaterialSpec {
public:
    typedef short int MatID_t;
    typedef gpuFloatType_t Density_t;

    static const MatID_t NULL_MATERIAL = std::numeric_limits<MatID_t>::min();

    MonteRay_MaterialSpec( void ) : ID( NULL_MATERIAL ), density( -1.0 ) {}

    MonteRay_MaterialSpec( MatID_t MatID, Density_t dens ) : ID( MatID ), density( dens ) {}

    MonteRay_MaterialSpec& operator=( const MonteRay_MaterialSpec& rhs ) {
        ID = rhs.ID;
        density = rhs.density;
        return *this;
    }

    MatID_t getID(void) const { return ID; }
    Density_t getDensity(void) const { return density; }

    void setID(MatID_t matid) { ID = matid; }
    void setDensity(Density_t d) { density = d; }

    void scaleDensity( Density_t d ) { density *= d; }

    bool operator==( const MonteRay_MaterialSpec& rhs ) {
        return ( this->getID() == rhs.getID()) && ( this->getDensity() == rhs.getDensity() );
    }

    bool operator!=( const MonteRay_MaterialSpec& rhs ) {
        return !( *this == rhs );
    }

    size_t bytesize(void) const {
        return sizeof( *this );
    }

private:
    MatID_t    ID;
    Density_t density;

};


// Note the following operators are global. This is done so that MaterialSpec
// scoping to the operator is not needed.

///Compares an MatID_t (or materialid) equals to MaterialSpec.ID
template<typename T>
bool operator==( T id, const MonteRay_MaterialSpec& matSpec ) {
    return ( MonteRay_MaterialSpec::MatID_t(id) == matSpec.getID() );
}

///Compares MaterialSpec.ID equals to an MatID_t (or materialid).
template<typename T>
bool operator==( const MonteRay_MaterialSpec& matSpec, T id ) {
    return ( id == matSpec );
}

///Compares an MatID_t (or materialid) not equal to MaterialSpec.ID
template<typename T>
bool operator!=( T id, const MonteRay_MaterialSpec& matSpec) {
    return !( id == matSpec );
}

///Compares MaterialSpec.ID not equal to an MatID_t (or materialid).
template<typename T>
bool operator!=( const MonteRay_MaterialSpec& matSpec, T id ) {
    return !( id == matSpec );
}

} // end namespace

#endif /*MONTERAY_MATERIALSPEC_HH_*/
