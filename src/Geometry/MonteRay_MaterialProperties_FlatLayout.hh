#ifndef MONTERAY_MATERIALPROPERTIES_FLATLAYOUT_HH_
#define MONTERAY_MATERIALPROPERTIES_FLATLAYOUT_HH_

#include <vector>
#include <functional>
#include <string>
#include <iostream>
#include <sstream>

#include "MonteRay_MaterialSpec.hh"
#include "MonteRay_CellProperties.hh"

namespace MonteRay {

///\brief Stores and retrieves properties such as material densities and IDs.

///\details Creates a material description container that holds a material list container\n
/// which holds the material specs (or the actually material properties like density and ID.)\n
/// To expand the list of property (ie, add temperature) then the MaterialSpec class has to be extented.

class MonteRay_MaterialProperties_FlatLayout {

public:
    typedef size_t offset_t;
    typedef int Cell_Index_t;
    typedef MonteRay::MonteRay_CellProperties::Material_Index_t Material_Index_t;
    typedef unsigned short Material_Index_Storage_t;

    typedef short int MatID_t;
    //typedef mcatk::CellProperties::MatID_t MatID_t;

    //typedef float Temperature_t;
    typedef MonteRay::MonteRay_CellProperties::Temperature_t Temperature_t;

    //typedef float Density_t;
    typedef MonteRay::MonteRay_CellProperties::Density_t Density_t;

public:
    ///Default Ctor
    MonteRay_MaterialProperties_FlatLayout(void) {}

    ///Ctor initialized by total number of cells.
    MonteRay_MaterialProperties_FlatLayout(std::size_t nCells);

    ///Ctor initialized by matIDs, density, and total number of cells (indexes).
    template<typename MaterialIDType, typename DensityType>
    MonteRay_MaterialProperties_FlatLayout(const std::vector<MaterialIDType>& IDs, const std::vector<DensityType>& dens, const std::size_t nCells){
        initializeMaterialDescription(IDs, dens, nCells );
    }

    ///Default Dtor
    ~MonteRay_MaterialProperties_FlatLayout(void) {};

    ///Set material description from another object, like lnk3dnt.
    template<typename objType>
    void setMaterialDescription(const objType& obj);

    ///Initializes the material description using vectors of matIDs and density.
    template<typename MaterialIDType, typename DensityType>
    void initializeMaterialDescription( const std::vector<MaterialIDType>& matid, const std::vector<DensityType>& dens, const std::size_t NTotalCells);

    template<typename MaterialIDType, typename DensityType, typename TempType>
    void initializeMaterialDescription( const std::vector<MaterialIDType>& IDs, const std::vector<DensityType>& dens, const std::vector<TempType>& temps, const std::size_t nCells);

    template<typename MaterialIDType, typename DensityType, typename TempType>
    void copyMaterialProperties( size_t nCells, size_t nMatSpecs, const size_t* pOffsetData, const TempType* pTemps, const MaterialIDType* pMatIDs, const DensityType* pDensities);

    offset_t getOffset( Cell_Index_t cellID ) const {
        if( singleNumComponents ) {
            return cellID * maxNumComponents;
        }
        return offset[ cellID ];
    }

    Material_Index_t getNumMaterials( Cell_Index_t cellID ) const {
        if( singleNumComponents ) { return maxNumComponents; }
        return offset[cellID+1] - offset[cellID];
    }

    MatID_t getMaterialID( Cell_Index_t cellID, Material_Index_t i ) const {
        return componentMatID[ getOffset(cellID) + i ];
    }

    void resetMaterialID( Cell_Index_t cellID, Material_Index_t i, MatID_t id ) {
        size_t location = getOffset(cellID) + i;
        if( i >= getNumMaterials(cellID) ) {
            std::stringstream msg;
            msg << "Exceeding the number of materials in the cell!\n";
            msg << "Cell index = " << cellID << ", material index = " << i << "\n";
            msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_MaterialProperties_FlatLayout::resetMaterialID" << "\n\n";
            throw std::runtime_error( msg.str() );
        }
        componentMatID[ getOffset(cellID) + i ] = id;
    }

    template<typename T>
    void renumberMaterialIDs(const T& matList ) {
        for( size_t i = 0; i < componentMatID.size(); ++i) {
            try{
                componentMatID[i] = matList.materialIDtoIndex( componentMatID[i] );
            }
            catch( const std::exception& e ) {
                std::stringstream msg;
                msg << e.what();
                msg << "Failure converting material ID to index!\n";
                msg << "Material ID = " << componentMatID[i] << ", component index = " << i << "\n";
                msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_MaterialProperties_FlatLayout::renumberMaterialIDs" << "\n\n";
                throw std::runtime_error( msg.str() );
            }
        }
    }

    Density_t getMaterialDensity( Cell_Index_t cellID, Material_Index_t i ) const {
        return componentDensity[ getOffset(cellID) + i ];
    }

    Temperature_t getTemperature( Cell_Index_t cellID) const {
        if( singleTemp ) { cellID = 0; }
        return temperature[ cellID ];
    }

    void add( MonteRay::MonteRay_CellProperties cell = MonteRay::MonteRay_CellProperties() );
    void addCellMaterial( Cell_Index_t cellID, MatID_t id, Density_t den );

    void removeMaterial( Cell_Index_t cellID, MatID_t id );

    ///Returns the size of the main container (MaterialDesc) in MonteRay_MaterialProperties_FlatLayout.
    size_t capacity(void) const { return offset.capacity(); }
    size_t size(void) const { return numCells; }
    Cell_Index_t getNTotalCells(void) const { return size(); }

    void scaleMaterialDensity( MatID_t ID, Density_t multiplier);
    void scaleAllMaterialDensities( Density_t multiplier);

    void setCellTemperature( const Cell_Index_t cellID, const Temperature_t temp);

    template <typename tempType>
    void setCellTemperatures( const std::vector<tempType>& temps);

    void setGlobalTemperature( Temperature_t tempMeV );

    void clear(void);
    void reserve( size_t NTotalCells, size_t nComponents );

    template<typename rangeType>
    void extract( const MonteRay_MaterialProperties_FlatLayout& A, const rangeType& obj);

    size_t bytesize(void) const;
    size_t capacitySize(void) const;
    size_t numEmptyMatSpecs(void) const;

    size_t getEqualNumMatMemorySize(Cell_Index_t nCells, Material_Index_Storage_t nMats ) const;
    size_t getNonEqualNumMatMemorySize(Cell_Index_t nCells, size_t nMatComponents ) const;

    /// returns a copy of a cell - not a reference
    MonteRay::MonteRay_CellProperties getCell( Cell_Index_t cellID ) const;

    bool isSingleTemp(void) const {
        return singleTemp;
    }

    bool isSingleNumMats(void) const { return singleNumComponents; }
    unsigned getMaxNumMats(void) const { return maxNumComponents; }

    size_t temperatureSize() const { return temperature.size(); }
    size_t offsetSize() const { return offset.size(); }
    size_t componentMatIDSize() const { return componentMatID.size(); }
    size_t componentDensitySize() const { return componentDensity.size(); }

    size_t temperatureCapacity() const { return temperature.capacity(); }
    size_t offsetCapacity() const { return offset.capacity(); }
    size_t componentMatIDCapacity() const { return componentMatID.capacity(); }
    size_t componentDensityCapacity() const { return componentDensity.capacity(); }

    void disableMemoryReduction();
    bool isMemoryReductionDisabled(void) const { return memoryReductionDisabled; }

    const offset_t* getOffsetData(void) const {return offset.data(); }
    const Temperature_t* getTemperatureData(void) const { return temperature.data(); }
    const MatID_t* getMaterialIDData(void) const { return componentMatID.data(); }
    const Density_t* getMaterialDensityData(void) const { return componentDensity.data(); }

protected:
    std::vector<offset_t> offset; ///< offset[cell] into component arrays with size of numCells+1
    std::vector<Temperature_t> temperature; ///< temperature[cell] with size of numCells

    /// componentMatID[offset+i] with size of totalNumMatComponents
    std::vector<MatID_t> componentMatID;

    /// componentDensity[offset+i] with size of totalNumMatComponents
    std::vector<Density_t> componentDensity;

    unsigned numCells = 0;
    unsigned numReservedCells = 0;
    unsigned totalNumComponents = 0;
    unsigned maxNumComponents = 0;
    bool singleTemp = true;
    bool singleNumComponents = false;
    bool memoryReductionDisabled = false;

    void convertFromSingleNumComponents(void);

};

///Set material description from another object, like lnk3dnt.
template<typename objType>
void
MonteRay_MaterialProperties_FlatLayout::setMaterialDescription(const objType& obj){
    size_t NTotalCells = obj.NCells(0) * obj.NCells(1) * obj.NCells(2);

    size_t NMaxMaterialsPerCell = obj.MaxMaterialsPerCell();
    std::vector< double > density( NTotalCells * NMaxMaterialsPerCell );
    std::vector< int >   material( NTotalCells * NMaxMaterialsPerCell );

    obj.fillDensityArray ( &density[0]  );
    obj.fillMaterialArray( &material[0] );

    initializeMaterialDescription( material, density, NTotalCells );
}

///Initializes the material description using vectors of matIDs and density.
template<typename MaterialIDType, typename DensityType>
void
MonteRay_MaterialProperties_FlatLayout::initializeMaterialDescription( const std::vector<MaterialIDType>& matid, const std::vector<DensityType>& dens, const std::size_t NTotalCells){
    if( matid.size() != dens.size() ) {
        std::stringstream msg;
        msg << "Material ID vector size is not the same as density vector size!\n";
        msg << "Material ID vector size = " << matid.size() << ", density vector size = " << dens.size() << "\n";
        msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_MaterialProperties_FlatLayout::initializeMaterialDescription" << "\n\n";
        throw std::runtime_error( msg.str() );
    }

    clear();

    // re-check disabled memory reductions after call to clear
    if( memoryReductionDisabled ) { disableMemoryReduction(); }

    MaterialIDType maxMatID = std::numeric_limits<MatID_t>::max();

    // get the max number of non-zero density materials in a cell
    maxNumComponents = 0;
    size_t nComponents = 0;
    for( size_t n=0; n<NTotalCells; ++n ) {
        unsigned numCellMats = 0;
        for( size_t index=n; index < matid.size(); index += NTotalCells ) {
            if( matid[index] <= MonteRay_MaterialSpec::NULL_MATERIAL || matid[index] > maxMatID ) {
                std::stringstream msg;
                msg << "Material ID exceeds MatID_t range!\n";
                msg << "Material ID = " << matid[index] << ", upper range limit = " <<  maxMatID << ", lower range limit = " << MonteRay_MaterialSpec::NULL_MATERIAL << "\n";
                msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_MaterialProperties_FlatLayout::initializeMaterialDescription" << "\n\n";
                std::cout << "MCATK Error: " << msg.str();
                throw std::runtime_error( msg.str() );
            }

            DensityType matdensity = dens[ index ];
            if( matdensity > 1.0e-10 ) {
                ++numCellMats;
                ++nComponents;
            }
        }
        if(numCellMats > maxNumComponents ) {
            maxNumComponents = numCellMats;
        }
    }

    size_t memorySizeForEqualNumMats = getEqualNumMatMemorySize( NTotalCells, maxNumComponents );
    size_t memorySizeForNonEqualNumMats = getNonEqualNumMatMemorySize( NTotalCells, nComponents );

    if( memorySizeForEqualNumMats < memorySizeForNonEqualNumMats && !memoryReductionDisabled ) {
        singleNumComponents = true;
        reserve( NTotalCells, NTotalCells * maxNumComponents );
    } else {
        singleNumComponents = false;
        reserve( NTotalCells, nComponents );
        offset.reserve( NTotalCells+1 );
    }

    MonteRay::MonteRay_CellProperties cell;
    cell.reserve( maxNumComponents );
    for( size_t n=0; n<NTotalCells; ++n ) {

        for( size_t index=n; index < matid.size(); index += NTotalCells ) {
            int ID = matid[ index ];
            DensityType matdensity = dens[ index ];
            if( matdensity > 1.0e-10 ) {
                cell.add( ID, matdensity );
            }
        }

        if( singleNumComponents ) {
            // fillup blank section of cell materials
            for( unsigned i=cell.size(); i<maxNumComponents; ++i ){
                cell.add( MonteRay_MaterialSpec::NULL_MATERIAL, 0.0 );
            }
        }

        add( cell );
        cell.clear();
    }
}

template<typename MaterialIDType, typename DensityType, typename TempType>
void
MonteRay_MaterialProperties_FlatLayout::initializeMaterialDescription( const std::vector<MaterialIDType>& IDs, const std::vector<DensityType>& dens, const std::vector<TempType>& temps, const std::size_t nCells){
    if( IDs.size() != temps.size() ) {
        std::stringstream msg;
        msg << "Material ID vector size is not the same as temperature vector size!\n";
        msg << "ID vector size = " << IDs.size() << ", temperature vector size = " << temps.size() << "\n";
        msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_MaterialProperties_FlatLayout::initializeMaterialDescription" << "\n\n";
        throw std::runtime_error( msg.str() );
    }
    initializeMaterialDescription( IDs, dens, nCells);
    setCellTemperatures(temps);
}

template<typename MaterialIDType, typename DensityType, typename TempType>
void
MonteRay_MaterialProperties_FlatLayout::copyMaterialProperties(
		size_t nCells, size_t nMatSpecs, const size_t* pOffsetData, const TempType* pTemps,
		const MaterialIDType* pMatIDs, const DensityType* pDensities)
{
	clear();
	numCells = nCells;
	totalNumComponents = nMatSpecs;
    numReservedCells = numCells;
    memoryReductionDisabled = true;

    offset.resize( nCells + 1 );
    for( unsigned i=0; i< nCells+1; ++i) {
    	offset[i] = pOffsetData[i];
    }

    temperature.resize( nCells );
    for( unsigned i=0; i< nCells; ++i) {
    	temperature[i] = pTemps[i];
    }

    componentMatID.resize( nMatSpecs );
    componentDensity.resize( nMatSpecs );
    for( unsigned i=0; i< nMatSpecs; ++i) {
    	componentMatID[i] = pMatIDs[i];
    	componentDensity[i] = pDensities[i];
    }
}

template<typename TempType>
void
MonteRay_MaterialProperties_FlatLayout::setCellTemperatures( const std::vector<TempType>& temps) {
    if( temps.size() != size() ) {
        std::stringstream msg;
        msg << "Temperature vector size is not the same size as the number of cells!\n";
        msg << "Number of cells = " << size() << ", temperature vector size = " << temps.size() << "\n";
        msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_MaterialProperties_FlatLayout::setCellTemperatures" << "\n\n";
        throw std::runtime_error( msg.str() );
    }
    singleTemp = true;
    temperature.clear();
    temperature.push_back( temps[0] );
    for( unsigned i=1; i< temps.size(); ++i ){
        setCellTemperature(i,temps[i]);
    }
}

template<typename rangeType>
void
MonteRay_MaterialProperties_FlatLayout::extract( const MonteRay_MaterialProperties_FlatLayout& A, const rangeType& obj) {
    clear();
    for(typename rangeType::const_iterator itr=obj.begin(); itr != obj.end(); ++itr) {
        add( A.getCell(*itr) );
    }
}

} // end namespace

#endif /* MONTERAY_MATERIALPROPERTIES_FLATLAYOUT_HH_ */
