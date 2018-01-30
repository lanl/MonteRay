#include "MonteRay_CellProperties.hh"

#include <sstream>
#include <stdexcept>

namespace MonteRay {

bool MonteRay_CellProperties::containsMaterial( MatID_t matID ) const {
    for( unsigned i=0; i<cellMaterials.size(); ++i ){
        if( cellMaterials[i].getID() == matID ){ return true; }
    }
    return false;
}

void MonteRay_CellProperties::scaleDensity(MatID_t id, Density_t density ) {
    for(Material_Index_t i=0; i < size(); ++i ){
        if( getMaterialID(i) == id ) {
            cellMaterials[i].scaleDensity( density );
            return;
        }
    }
    std::stringstream msg;
    msg << "Material id not found!\n";
    msg << "Called from : " << __FILE__ << " [" << __LINE__ << "] : " << "MonteRay_CellProperties::scaleDensity" << "\n\n";
    throw std::runtime_error( msg.str() );
}

void MonteRay_CellProperties::changeDensity(MatID_t id, Density_t density ) {
    for(Material_Index_t i=0; i < size(); ++i ){
        if( getMaterialID(i) == id ) {
            cellMaterials[i].setDensity( density );
            return;
        }
    }

    std::stringstream msg;
    msg << "Material id not found!\n";
    msg << "Called from : " << __FILE__ << " [" << __LINE__ << "] : " << "MonteRay_CellProperties::changeDensity" << "\n\n";
    throw std::runtime_error( msg.str() );
}

void MonteRay_CellProperties::scaleAllDensities(Density_t density) {
    for(Material_Index_t i=0; i < size(); ++i ){
        cellMaterials[i].scaleDensity( density );
    }
}

void MonteRay_CellProperties::removeMaterial( MatID_t id ) {

    typename MatList_t::iterator pos = cellMaterials.begin();
    // Adjust density if the material is already a part of the cell
    while( pos->getID() != id && pos != cellMaterials.end() ) {
        ++pos;
    }

    if( pos != cellMaterials.end() ) {
        cellMaterials.erase( pos );
        return;
    }

    std::stringstream msg;
    msg << "Material id not found!\n";
    msg << "Called from : " << __FILE__ << " [" << __LINE__ << "] : " << "MonteRay_CellProperties::removeMaterial" << "\n\n";
    throw std::runtime_error( msg.str() );
}


size_t MonteRay_CellProperties::bytesize(void) const {
    size_t total = sizeof( *this );
    for( size_t i =0; i < size(); ++i ) {
        total += cellMaterials[i].bytesize();
    }
    return total;
}

size_t MonteRay_CellProperties::capacitySize(void) const {
    size_t total = bytesize();
    size_t materialSize = sizeof( MonteRay_MaterialSpec );
    for( size_t i = size(); i < capacity(); ++i ) {
        total += materialSize;
    }
    return total;
}

size_t MonteRay_CellProperties::numEmptyMatSpecs(void) const {
    size_t total = 0;
    for( size_t i = size(); i < capacity(); ++i ) {
        ++total;
    }
    return total;
}

void MonteRay_CellProperties::shrink_to_fit(void) {
    cellMaterials.shrink_to_fit();
}

void MonteRay_CellProperties::reserve( size_t num) {
    cellMaterials.reserve( num );
}

} /* namespace mcatk */
