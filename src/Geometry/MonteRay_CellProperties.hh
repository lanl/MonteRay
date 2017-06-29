#ifndef MONTERAY_CELLPROPERTIES_HH_
#define MONTERAY_CELLPROPERTIES_HH_

#include <vector>
#include <cassert>

#include "MonteRay_MaterialSpec.hh"

namespace MonteRay {

class MonteRay_CellProperties {

public:
    typedef unsigned Material_Index_t;
    typedef double Temperature_t;
    typedef MonteRay_MaterialSpec::MatID_t MatID_t;
    typedef MonteRay_MaterialSpec::Density_t Density_t;

private:
    typedef std::vector< MonteRay_MaterialSpec > MatList_t;

public:
    MonteRay_CellProperties(void) : cellTemperature(-99.0){};
    ~MonteRay_CellProperties(void){};

    void operator=( const MatList_t& ml ) { cellMaterials = ml; }

    void setTemperature( Temperature_t t){ cellTemperature = t; }
    Temperature_t getTemperature( void ) const { return cellTemperature; }

    bool containsMaterial( MatID_t matID ) const;

    size_t capacity(void) const { return cellMaterials.capacity(); }
    size_t size(void) const { return cellMaterials.size(); }
    Material_Index_t getNumMaterials(void) const { return size(); }

    Material_Index_t getMaterialID( Material_Index_t i ) const {
        assert( i < size() );
        return cellMaterials[i].getID();
    }

    Density_t getMaterialDensity( Material_Index_t i ) const {
        assert( i < size() );
        return cellMaterials[i].getDensity();
    }

    void add( MonteRay_MaterialSpec ms ) { cellMaterials.push_back( ms ); }
    void add( MatID_t matID, Density_t density ) { cellMaterials.push_back( MonteRay_MaterialSpec(matID, density) ); }
    void clear(void) { cellMaterials.clear(); }

    void scaleDensity(MatID_t id, Density_t density );
    void changeDensity(MatID_t id, Density_t density );
    void scaleAllDensities(Density_t density);
    void removeMaterial( MatID_t id );

    template<typename FUNC_T, typename T = double >
    T getXsecSum( FUNC_T& func ) const;

    template<typename FUNC_T, typename T = double >
    T getXsecSumPerMass( FUNC_T& func ) const;

    size_t bytesize(void) const;
    size_t capacitySize(void) const;
    size_t numEmptyMatSpecs(void) const;
    void shrink_to_fit(void);
    void reserve( size_t num);

private:
    Temperature_t cellTemperature;
    MatList_t cellMaterials;
};

template<typename FUNC_T, typename T >
T MonteRay_CellProperties::getXsecSum( FUNC_T& func) const {
    Temperature_t temp = getTemperature();

    T sum = T(0);
    for( Material_Index_t material=0; material < size(); ++material ) {
        unsigned ID    = getMaterialID(material);
        double density = getMaterialDensity(material);

        sum += func( ID, density, temp );
    }
    return sum;
}

template<typename FUNC_T, typename T >
T MonteRay_CellProperties::getXsecSumPerMass( FUNC_T& func) const {
    Temperature_t temp = getTemperature();

    T sum = T(0);
    for( Material_Index_t material=0; material < size(); ++material ) {
        unsigned ID    = getMaterialID(material);
        double density = getMaterialDensity(material);
        if( density > 0.0 ) {
            sum += func( ID, density, temp ) / density;
        }
    }
    return sum;
}

} /* namespace MonteRay */

#endif /* MONTERAY_CELLPROPERTIES_HH_ */
