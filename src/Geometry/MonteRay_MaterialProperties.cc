#include "MonteRay_MaterialProperties.hh"

namespace MonteRay{

void 
MonteRay_MaterialProperties::setCellTemperatureCelsius( const Cell_Index_t cell, const Temperature_t temperatureCelsius){
    std::stringstream msg;
    msg << "Disabled in MonteRay!\n";
    msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_MaterialProperties::setCellTemperatureCelsius" << "\n\n";
    throw std::runtime_error( msg.str() );
//    checkCellIndex( cell, __FILE__, __LINE__);
//    double temp = (temperatureCelsius + 273.15)  / mcatk::Constants::MeVtoKelvin;
//    setCellTemperature(cell, temp);
}

MonteRay_MaterialProperties::Temperature_t
MonteRay_MaterialProperties::getTemperatureCelsius( Cell_Index_t cellID) const {
    std::stringstream msg;
    msg << "Disabled in MonteRay!\n";
    msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_MaterialProperties::getTemperatureCelsius" << "\n\n";
    throw std::runtime_error( msg.str() );
//    checkCellIndex( cellID, __FILE__, __LINE__);
//    double temp = getTemperature(cellID);
//    return temp * mcatk::Constants::MeVtoKelvin - 273.15;
}

void
MonteRay_MaterialProperties::add( MonteRay::MonteRay_CellProperties cell ) {
    pMemoryLayout->add( cell );
}

void
MonteRay_MaterialProperties::addCellMaterial( Cell_Index_t cellID, MatID_t id, Density_t den ) {
    forceCheckCellIndex( cellID );
    if( containsMaterial(cellID, id) ) {
        std::stringstream msg;
        msg << "Unable to add material to cell. Material already exists!\n";
        msg << "cell index = " << cellID << ", material ID = " << id << "\n";
        msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "'MonteRay_MaterialProperties::addCellMaterial" << "\n\n";
        throw std::runtime_error( msg.str() );
    }
    pMemoryLayout->addCellMaterial( cellID, id, den );
}

void
MonteRay_MaterialProperties::removeMaterial( Cell_Index_t cellID, MatID_t id ) {
    checkCellIndex( cellID, __FILE__, __LINE__);
    pMemoryLayout->removeMaterial( cellID, id );
}

/// Provides a safe external method for returning the material ID by cell, checks for null material
MonteRay_MaterialProperties::MatID_t
MonteRay_MaterialProperties::getMaterialID( Cell_Index_t cellID, Material_Index_t i ) const {
    MatID_t ID = getMaterialIDNotSafe(cellID, i);

    if( ID == MonteRay_MaterialSpec::NULL_MATERIAL ) {
        std::stringstream msg;
        msg << "Returning NULL_MATERIAL material ID, avoid external call to getMaterialID, call getFuncSumByCell instead!\n";
        msg << "cell index = " << cellID << ", material index = " << i << "\n";
        msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_MaterialProperties::getMaterialID" << "\n\n";
        throw std::runtime_error( msg.str() );
    }
    return ID;
}

} /* End namespace MonteRay */
