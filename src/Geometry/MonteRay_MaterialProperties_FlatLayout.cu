#include "MonteRay_MaterialProperties_FlatLayout.hh"

namespace MonteRay {

MonteRay_MaterialProperties_FlatLayout::MonteRay_MaterialProperties_FlatLayout(std::size_t nCells) {
    reserve(nCells, nCells);
    for( unsigned i=0; i< nCells; ++i ) {
        add();
    }
}

void MonteRay_MaterialProperties_FlatLayout::add( MonteRay::MonteRay_CellProperties cell ) {
    ++numCells;

    Material_Index_Storage_t numCellComponents = cell.getNumMaterials();

    unsigned currentCell = numCells-1;
    if( temperature.empty()  ) {
        temperature.push_back( cell.getTemperature() );
    } else {
        setCellTemperature( currentCell, cell.getTemperature() );
    }

    if( !singleNumComponents ) {
        if( offset.empty() ) {
            offset.push_back( 0 );
            offset.push_back( numCellComponents );
        } else {
            offset.push_back( offset[currentCell] + numCellComponents );
        }
    }

    for( Material_Index_Storage_t i=0; i< numCellComponents; ++i) {
        componentMatID.push_back( cell.getMaterialID(i) );
        componentDensity.push_back( cell.getMaterialDensity(i) );
    }
}

void
MonteRay_MaterialProperties_FlatLayout::convertFromSingleNumComponents(void) {
    unsigned numMats = getNumMaterials(0);

    offset.clear();
    offset.reserve( numCells+1 );

    for( unsigned i=0; i< numCells+1; ++i ) {
        offset.push_back( i*numMats );
    }
    singleNumComponents = false;
}

void MonteRay_MaterialProperties_FlatLayout::setCellTemperature( Cell_Index_t cellID, Temperature_t temp  ) {
    // keeps a single temperature value until a differing temperature is added,
    // then converts the single temperature into an array.
    // single global temperature is stored in temperature[0]

    assert( cellID < numCells );

    if( singleTemp ) {
        double globalTemp = temperature[0];
        if( temp != globalTemp ) {
            // convert to multi-temperature
            singleTemp = false;

            temperature.reserve( std::max(numReservedCells,numCells) );
            for( unsigned i=1; i<numCells; ++i) {
                temperature.push_back( globalTemp );
            }
        }
    }

    if( !singleTemp ) {

        if( cellID < temperature.size() ) {
            temperature[cellID] = temp;
        } else if( cellID == temperature.size() ){
            if( temperature.size() == 1 ) {
                temperature.reserve( std::max(numReservedCells,numCells) );
            }
            temperature.push_back( temp );
        } else {
            std::stringstream msg;
            msg << "When setting cell temperatures, the cell must be added first!\n";
            msg << "Cell index = " << cellID << ", temperature = " << temp << "\n";
            msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_MaterialProperties_FlatLayout::setCellTemperature" << "\n\n";
            std::cout << "MCATK Error: " << msg.str();
            throw std::runtime_error( msg.str() );
        }
    }
}

void MonteRay_MaterialProperties_FlatLayout::addCellMaterial( Cell_Index_t cellID, MatID_t id, Density_t den ){
    if( singleNumComponents ) {
        convertFromSingleNumComponents();
    }

    size_t insertAt = offset[cellID] + getNumMaterials(cellID);

    auto ID_itr = componentMatID.begin();
    componentMatID.insert(ID_itr+insertAt, id);

    auto Den_itr = componentDensity.begin();
    componentDensity.insert(Den_itr+insertAt, den);

    for( unsigned i=cellID+1; i<size()+1; ++i ){
        ++offset[i];
    }
}

void
MonteRay_MaterialProperties_FlatLayout::removeMaterial( Cell_Index_t cellID, MatID_t id ) {
    if( singleNumComponents ) {
        convertFromSingleNumComponents();
    }
    for( unsigned i=0; i<getNumMaterials(cellID); ++i) {
        if( getMaterialID(cellID,i) == id ) {
            size_t removeAt = offset[cellID] + i;

            auto ID_itr = componentMatID.begin();
            componentMatID.erase(ID_itr+removeAt);

            auto Den_itr = componentDensity.begin();
            componentDensity.erase(Den_itr+removeAt);

            for( unsigned i=cellID+1; i<size()+1; ++i ){
                --offset[i];
            }

            return;
        }
    }

    std::stringstream msg;
    msg << "Cell does not contain material id - CAN'T remove a material that doesn't exist!\n";
    msg << "Cell index = " << cellID << ", material ID = " << id << "\n";
    msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_MaterialProperties_FlatLayout::removeMaterial" << "\n\n";
    std::cout << "MCATK Error: " << msg.str();
    throw std::runtime_error( msg.str() );
}

void
MonteRay_MaterialProperties_FlatLayout::scaleMaterialDensity( MatID_t ID, Density_t multiplier) {
    for( unsigned i=0; i< componentMatID.size(); ++i ) {
       if( componentMatID[i] == ID ) {
           componentDensity[i] *= multiplier;
       }
    }
}

void
MonteRay_MaterialProperties_FlatLayout::scaleAllMaterialDensities( Density_t multiplier) {
    for( unsigned i=0; i< componentMatID.size(); ++i ) {
        componentDensity[i] *= multiplier;
    }
}

void
MonteRay_MaterialProperties_FlatLayout::setGlobalTemperature( Temperature_t tempMeV ) {
    temperature.resize(1);
    temperature[0] = tempMeV;
    singleTemp = true;
}

void
MonteRay_MaterialProperties_FlatLayout::clear(void) {
    offset.clear();
    temperature.clear();
    componentMatID.clear();
    componentDensity.clear();

    numCells = 0;
    numReservedCells = 0;
    totalNumComponents = 0;
    singleTemp = true;
    singleNumComponents = false;
}

void
MonteRay_MaterialProperties_FlatLayout::reserve( size_t NTotalCells, size_t nComponents ) {
    numReservedCells = NTotalCells;
    totalNumComponents = nComponents;

    if( memoryReductionDisabled ) {
        temperature.reserve( NTotalCells );
        offset.reserve(NTotalCells+1);
    }
    componentMatID.reserve( nComponents );
    componentDensity.reserve( nComponents );
}

size_t
MonteRay_MaterialProperties_FlatLayout::bytesize(void) const {
    size_t total = sizeof(*this);
    total += sizeof(offset_t)*offset.size();
    total += sizeof(Temperature_t)*temperature.size();
    total += sizeof(MatID_t)*componentMatID.size();
    total += sizeof(Density_t)*componentDensity.size();
    return total;
}

size_t
MonteRay_MaterialProperties_FlatLayout::capacitySize(void) const {
    size_t total = sizeof(*this);
    total += sizeof(offset_t)*offset.capacity();
    total += sizeof(Temperature_t)*temperature.capacity();
    total += sizeof(MatID_t)*componentMatID.capacity();
    total += sizeof(Density_t)*componentDensity.capacity();
    return total;
}

size_t
MonteRay_MaterialProperties_FlatLayout::getNonEqualNumMatMemorySize(Cell_Index_t nCells, size_t nMatComponents) const{
    size_t total = 0;
    total += sizeof(offset_t)*(nCells+1);
    total += sizeof(MatID_t)*nMatComponents;
    total += sizeof(Density_t)*nMatComponents;
    return total;
}

size_t
MonteRay_MaterialProperties_FlatLayout::getEqualNumMatMemorySize(Cell_Index_t nCells, Material_Index_Storage_t nMats ) const{
    size_t total = 0;
    total += sizeof(MatID_t)*nMats*nCells;
    total += sizeof(Density_t)*nMats*nCells;
    return total;
}


size_t
MonteRay_MaterialProperties_FlatLayout::numEmptyMatSpecs(void) const {
    return componentMatID.capacity() - componentMatID.size();
}

MonteRay::MonteRay_CellProperties
MonteRay_MaterialProperties_FlatLayout::getCell( Cell_Index_t cellID ) const {
    assert( cellID > -1 );
    assert( cellID < size() );

    MonteRay::MonteRay_CellProperties cell;
    cell.setTemperature( getTemperature(cellID) );

    for( unsigned i=0; i<getNumMaterials(cellID); ++i) {
        MatID_t ID = getMaterialID(cellID,i);
        if( ID == MonteRay_MaterialSpec::NULL_MATERIAL ) break;

        Density_t density = getMaterialDensity(cellID,i);
        cell.add( ID, density );
    }
    return std::move(cell);
}

void
MonteRay_MaterialProperties_FlatLayout::disableMemoryReduction() {
    memoryReductionDisabled = true;
    singleTemp = false;
    singleNumComponents = false;

    if( size() > 0 ) {
        std::stringstream msg;
        msg << "Disable memory reduction called after cells have already been added!\n";
        msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : MonteRay_MaterialProperties_FlatLayout::disableMemoryReduction\n\n";
        throw std::runtime_error( msg.str() );
    }
}
} // end namespace
