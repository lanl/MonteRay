#ifndef MONTERAY_MATERIALPROPERTIES_HH_
#define MONTERAY_MATERIALPROPERTIES_HH_

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <ostream>
#include <fstream>
#include <memory>

#include "MonteRay_MaterialProperties_FlatLayout.hh"
#include "MonteRayTypes.hh"

namespace MonteRay{

struct MonteRay_MaterialProperties_Data {
    typedef size_t offset_t;
    typedef MonteRay_CellProperties::Material_Index_t Material_Index_t;
    typedef MonteRay_CellProperties::Temperature_t Temperature_t;
    typedef MonteRay_CellProperties::MatID_t MatID_t;
    typedef MonteRay_CellProperties::Density_t Density_t;

    size_t numCells;
    size_t numMaterialComponents;

    offset_t* offset;
    MatID_t* ID;
    Density_t* density;
};

void ctor(MonteRay_MaterialProperties_Data*, unsigned num );
void dtor(MonteRay_MaterialProperties_Data* );

void cudaCtor(struct MonteRay_MaterialProperties_Data*,struct MonteRay_MaterialProperties_Data*);
void cudaDtor(struct MonteRay_MaterialProperties_Data*);

CUDA_CALLABLE_MEMBER
size_t getNumCells(const struct MonteRay_MaterialProperties_Data* ptr );

CUDA_CALLABLE_MEMBER
MonteRay_MaterialProperties_Data::offset_t getNumMats(const struct MonteRay_MaterialProperties_Data* ptr, unsigned i );

CUDA_CALLABLE_MEMBER
MonteRay_MaterialProperties_Data::Density_t getDensity(const struct MonteRay_MaterialProperties_Data* ptr, unsigned cellNum, unsigned matNum );

CUDA_CALLABLE_MEMBER
MonteRay_MaterialProperties_Data::MatID_t getMatID(const struct MonteRay_MaterialProperties_Data* ptr, unsigned cellNum, unsigned matNum );

CUDA_CALLABLE_KERNEL void kernelGetNumCells(MonteRay_MaterialProperties_Data* mp, unsigned* results );

CUDA_CALLABLE_KERNEL void kernelGetNumMaterials(MonteRay_MaterialProperties_Data* mp, unsigned cellNum, MonteRay_MaterialProperties_Data::Material_Index_t* results );

CUDA_CALLABLE_KERNEL void kernelGetMaterialID(MonteRay_MaterialProperties_Data* mp, unsigned cellNum, unsigned i, MonteRay_MaterialProperties_Data::MatID_t* results );

CUDA_CALLABLE_KERNEL void kernelGetMaterialDensity(MonteRay_MaterialProperties_Data* mp, unsigned cellNum, unsigned i, MonteRay_MaterialProperties_Data::Density_t* results );

CUDA_CALLABLE_KERNEL void kernelSumMatDensity(MonteRay_MaterialProperties_Data* mp, MonteRay_MaterialProperties_Data::MatID_t matIndex, MonteRay_MaterialProperties_Data::Density_t* results );

///\brief Stores and retrieves properties such as material densities and IDs.

///\details Provides an interface to the underlying memory storage.
///         Class does not hold data, only pointer to storage layout.
///         May change in the future when one storage layout is decided upon.

class MonteRay_MaterialProperties {

public:

    //typedef MaterialProperties_StandardLayout MemoryLayout_t;
    typedef MonteRay::MonteRay_MaterialProperties_FlatLayout MemoryLayout_t;
    typedef std::unique_ptr<MemoryLayout_t> pMemoryLayout_t;

    typedef int Cell_Index_t;
    typedef size_t offset_t;
    typedef MonteRay::MonteRay_CellProperties::Material_Index_t Material_Index_t;
    typedef MonteRay::MonteRay_CellProperties::Temperature_t Temperature_t;
    typedef MonteRay::MonteRay_CellProperties::MatID_t MatID_t;
    typedef MonteRay::MonteRay_CellProperties::Density_t Density_t;

public:
    ///Default Ctor
    MonteRay_MaterialProperties(void) {
        pMemoryLayout.reset( new MemoryLayout_t() );
        disableMemoryReduction();
        ptrData = new MonteRay_MaterialProperties_Data;
    }

    ///Ctor initialized by total number of cells.
    MonteRay_MaterialProperties(const std::size_t& nCells) {
        pMemoryLayout.reset( new MemoryLayout_t(nCells) );
        disableMemoryReduction();
        ptrData = new MonteRay_MaterialProperties_Data;
    }

    ///Ctor initialized by matIDs, density, and total number of cells (indexes).
    template<typename MaterialIDType, typename DensityType>
    MonteRay_MaterialProperties(const std::vector<MaterialIDType>& IDs, const std::vector<DensityType>& dens, const std::size_t nCells){
        pMemoryLayout.reset( new MemoryLayout_t(IDs, dens, nCells) );
        ptrData = new MonteRay_MaterialProperties_Data;
    }

    ///Default Dtor
    virtual ~MonteRay_MaterialProperties(void) {
        cudaDtor();
        delete ptrData;
    }

    void cudaDtor(void);
    void copyToGPU(void);

    ///Set material description from another object, like lnk3dnt.
    template<typename objType>
    void setMaterialDescription(const objType& obj);

    ///Set material description from another object, like lnk3dnt with view.
    template<typename objType, typename iter>
    void setMaterialDescription( objType& obj, const iter& view );

    ///Initializes the material description using vectors of matIDs and density.
    template<typename MaterialIDType, typename DensityType>
    void initializeMaterialDescription( const std::vector<MaterialIDType>& IDs, const std::vector<DensityType>& dens, const std::size_t nCells) {
        pMemoryLayout->template initializeMaterialDescription<MaterialIDType,DensityType>( IDs, dens, nCells );
        setupPtrData();
    }

    template<typename MaterialIDType, typename DensityType, typename TempType>
    void initializeMaterialDescription( const std::vector<MaterialIDType>& IDs, const std::vector<DensityType>& dens, const std::vector<TempType>& temps, const std::size_t nCells) {
        pMemoryLayout->template initializeMaterialDescription<MaterialIDType,DensityType,TempType>( IDs, dens, temps, nCells );
        setupPtrData();
    }

    /// initializer that expects an object like MaterialProperties
    template<typename MATPROPS_T>
    void copyMaterialProperties( MATPROPS_T& matprops  ) {
        if( ! matprops.isMemoryReductionDisabled() ) {
            throw std::runtime_error("MonteRay_MaterialProperties::copyMaterialProperties -- original material properties must have memory reduction disabled.");
        }
        using MAT_T = typename MATPROPS_T::MatID_t;
        using DEN_T = typename MATPROPS_T::Density_t;
        using TEMP_T = typename MATPROPS_T::Temperature_t;
        pMemoryLayout->template copyMaterialProperties<MAT_T, DEN_T, TEMP_T >( matprops.size(),
                matprops.numMatSpecs(),
                matprops.getOffsetData(),
                matprops.getTemperatureData(),
                matprops.getMaterialIDData(),
                matprops.getMaterialDensityData()
        );
        setupPtrData();
    }

    void setupPtrData(void) {
        ptrData->numCells = size();
        ptrData->numMaterialComponents = numMatSpecs();
        ptrData->offset = const_cast<offset_t*>(getOffsetData());
        ptrData->ID = const_cast<MatID_t*>(getMaterialIDData());
        ptrData->density = const_cast<Density_t*>(getMaterialDensityData());
    }

    template<typename T>
    void setupPtrData(const T& obj ) {
        ptrData->numCells = obj.size();
        ptrData->numMaterialComponents = obj.numMatSpecs();
        ptrData->offset = const_cast<offset_t*>(obj.getOffsetData());
        ptrData->ID = const_cast<MatID_t*>(obj.getMaterialIDData());
        ptrData->density = const_cast<Density_t*>(obj.getMaterialDensityData());
    }

    template< typename FUNC_T, typename CELLINDEX_T, typename T = double >
    T getFuncSumByCell( FUNC_T& func, CELLINDEX_T cell) const;

    template<typename FUNC_T, typename SomeParticle_t, typename T = double >
    T getXsecSum( FUNC_T& func, const SomeParticle_t& p) const;

    template<typename FUNC_T, typename SomeParticle_t, typename T = double >
    T getXsecSumPerMass( FUNC_T& func, const SomeParticle_t& p) const;

    template<typename T>
    using SomeClass_t = typename std::enable_if<std::is_class<T>::value, T>::type;

    template<typename SomeParticle_t>
    Material_Index_t getNumMaterials( const SomeClass_t<SomeParticle_t> &p ) const {
        return getNumMaterials( getCellIndex(p) );
    }

    Material_Index_t getNumMaterials( Cell_Index_t cellID ) const {
        checkCellIndex( cellID, __FILE__, __LINE__);
        return pMemoryLayout->getNumMaterials( cellID );
    }

    /// Provides an external method for returning the material ID by particle location, checks for null material
    template<typename SomeParticle_t>
    MatID_t getMaterialID( const SomeClass_t<SomeParticle_t> &p, Material_Index_t i ) const {
        return getMaterialID( getCellIndex(p), i);
    }

    /// Provides a safe external method for returning the material ID by cell, 
    /// checks for null material
    MatID_t getMaterialID( Cell_Index_t cellID, Material_Index_t i ) const;

private:
    /// Provides an internal method for returning the material ID by cell, doesn't check for null material
    MatID_t getMaterialIDNotSafe( Cell_Index_t cellID, Material_Index_t i ) const {
        checkCellIndex( cellID, __FILE__, __LINE__);
        assert( i < getNumMaterials(cellID) );
        return pMemoryLayout->getMaterialID( cellID, i );
    }

public:
    template<typename SomeParticle_t>
    Density_t getMaterialDensity( const SomeClass_t<SomeParticle_t> &p, Material_Index_t i ) const {
        return getMaterialDensity( getCellIndex(p), i);
    }

    Density_t getMaterialDensity( Cell_Index_t cellID, Material_Index_t i ) const {
        checkCellIndex( cellID, __FILE__, __LINE__);
        assert( i < getNumMaterials(cellID) );
        return pMemoryLayout->getMaterialDensity( cellID, i );
    }

    template<typename SomeParticle_t>
    Temperature_t getTemperature(  const SomeClass_t<SomeParticle_t> &p ) const {
        return getTemperature( getCellIndex(p) );
    }

    Temperature_t getTemperature( Cell_Index_t cellID) const {
        checkCellIndex( cellID, __FILE__, __LINE__);
        return pMemoryLayout->getTemperature( cellID );
    }

    bool containsMaterial( Cell_Index_t cellID, MatID_t id ) const {
        checkCellIndex( cellID, __FILE__, __LINE__);
        for( unsigned i=0; i<getNumMaterials(cellID); ++i) {
            if( getMaterialIDNotSafe(cellID,i) == id ) { return true; }
        }
        return false;
    }

    template<typename T>
    void renumberMaterialIDs(const T& matList ) {
        pMemoryLayout->renumberMaterialIDs<T>(matList);
    }

    void add( MonteRay::MonteRay_CellProperties cell = MonteRay::MonteRay_CellProperties() );
    void addCellMaterial( Cell_Index_t cellID, MatID_t id, Density_t den );
    void removeMaterial( Cell_Index_t cellID, MatID_t id );

    ///Returns the size of the main container (MaterialDesc) in MonteRay_MaterialProperties.
    size_t capacity(void) const { return pMemoryLayout->capacity(); }
    size_t size(void) const { return pMemoryLayout->size(); }
    Cell_Index_t getNTotalCells(void) const { return size(); }

    void scaleMaterialDensity( MatID_t ID, Density_t den){
        pMemoryLayout->scaleMaterialDensity(ID,den);
    }

    void scaleAllMaterialDensities( Density_t den ){
        pMemoryLayout->scaleAllMaterialDensities(den);
    }

    void setCellTemperature( const Cell_Index_t cell, const Temperature_t temperature){
        checkCellIndex( cell, __FILE__, __LINE__);
        pMemoryLayout->setCellTemperature(cell,temperature);
    }

    template <typename tempType>
    void setCellTemperatures( const std::vector<tempType>& temperatures) {
        pMemoryLayout->setCellTemperatures(temperatures);
    }

    void setGlobalTemperature( Temperature_t tempMeV ) {
        pMemoryLayout->setGlobalTemperature(tempMeV);
    }

    void setCellTemperatureCelsius( const Cell_Index_t cell, const Temperature_t temperatureCelsius);
    Temperature_t getTemperatureCelsius( Cell_Index_t cellID) const;

    void clear(void) { pMemoryLayout->clear(); }

    template<typename rangeType>
    void extract( const MonteRay_MaterialProperties& A, const rangeType& obj) {
        pMemoryLayout->template extract<rangeType>( *(A.pMemoryLayout) ,obj);
    }

    /// returns cell by copy - underlying data may not be a cell so can't return a reference
    MonteRay::MonteRay_CellProperties getCell( Cell_Index_t cellID ) const {
        checkCellIndex( cellID, __FILE__, __LINE__);
        return pMemoryLayout->getCell(cellID);
    }

    size_t bytesize(void) const { return pMemoryLayout->bytesize(); }
    size_t capacitySize(void) const { return pMemoryLayout->capacitySize(); }
    size_t numEmptyMatSpecs(void) const { return pMemoryLayout->numEmptyMatSpecs(); }
    size_t numMatSpecs(void) const { return pMemoryLayout->componentMatIDSize(); }

    const offset_t* getOffsetData(void) const { return pMemoryLayout->getOffsetData(); }
    const Temperature_t* getTemperatureData(void) const { return pMemoryLayout->getTemperatureData(); }
    const MatID_t* getMaterialIDData(void) const { return pMemoryLayout->getMaterialIDData(); }
    const Density_t* getMaterialDensityData(void) const { return pMemoryLayout->getMaterialDensityData(); }

    void disableMemoryReduction() { pMemoryLayout->disableMemoryReduction(); }
    bool isMemoryReductionDisabled(void) const { return pMemoryLayout->isMemoryReductionDisabled(); }

    Density_t launchSumMatDensity(MatID_t matIndex) const;
    Density_t sumMatDensity( MatID_t matIndex) const;

    size_t launchGetNumCells(void) const;
    Material_Index_t launchGetNumMaterials( Cell_Index_t cellID ) const;
    MatID_t launchGetMaterialID( Cell_Index_t cellID, Material_Index_t i ) const;
    Density_t launchGetMaterialDensity( Cell_Index_t cellID, Material_Index_t i ) const;

    const MonteRay_MaterialProperties_Data* getPtr( void ) const {
        return ptrData;
    }
private:
    pMemoryLayout_t pMemoryLayout;
    MonteRay_MaterialProperties_Data* ptrData = NULL;
    MonteRay_MaterialProperties_Data* tempData = NULL;
    bool cudaCopyMade = false;

public:
    MonteRay_MaterialProperties_Data* ptrData_device = NULL;

private:

    void checkCellIndex(Cell_Index_t index, const char* fname, int line ) const {
#ifndef NDEBUG
        if( index >= size() ) {
            std::stringstream msg;
            msg << "Error: File= " << std::string(fname) <<":"<< line <<  " -- Cell index is greater than the size of the number of cells.";
            std::cout << msg.str() << std::endl;
            throw std::runtime_error( msg.str() );
        }
        if( index < 0 ) {
            std::stringstream msg;
            msg << "Error: File= " << std::string(fname) <<":"<< line <<  " -- Cell index is negative.";
            std::cout << msg.str() << std::endl;
            throw std::runtime_error( msg.str() );
        }
#endif
    }

    void forceCheckCellIndex(Cell_Index_t index) const {
#ifndef NDEBUG
        checkCellIndex( index, __FILE__, __LINE__);
#else
        if( index < 0 || index >= size() ) {
            std::stringstream msg;
            msg << "Invalid cell index!  index =" << index << ", size=" << size() << "\n";
            msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_MaterialProperties::forceCheckCellIndex" << "\n\n";
            throw std::runtime_error( msg.str() );
        }
#endif
    }

protected:
    template<typename SomeParticle_t>
    Cell_Index_t getCellIndex(const SomeParticle_t& p) const {
        Cell_Index_t index = p.getLocationIndex();
        if( index == -1 ) { index = 0; } // Solid 3D GeomNode w/o mesh
        checkCellIndex( index, __FILE__, __LINE__);
        return index;
    }

};

///Set material description from another object, like lnk3dnt.
template<typename objType>
void
MonteRay_MaterialProperties::setMaterialDescription(const objType& obj){
    size_t NTotalCells = obj.NCells(0) * obj.NCells(1) * obj.NCells(2);

    size_t NMaxMaterialsPerCell = obj.MaxMaterialsPerCell();
    std::vector< double > density( NTotalCells * NMaxMaterialsPerCell );
    std::vector< int >   material( NTotalCells * NMaxMaterialsPerCell );

    obj.fillDensityArray ( &density[0]  );
    obj.fillMaterialArray( &material[0] );

    initializeMaterialDescription( material, density, NTotalCells );
}

///Set material description from another object, like lnk3dnt with view.
template<typename objType, typename iter>
void
MonteRay_MaterialProperties::setMaterialDescription( objType& obj, const iter& view ){
    std::vector< double > density;
    std::vector< int >    material;
    obj.extract(material, density, view );
    initializeMaterialDescription( material, density, view.size() );
}

template< typename FUNC_T, typename CELLINDEX_T, typename T >
T
MonteRay_MaterialProperties::getFuncSumByCell( FUNC_T& func, CELLINDEX_T cellID) const{
    Temperature_t temp = getTemperature(cellID);

    T sum = T(0);
    for( Material_Index_t material=0; material < getNumMaterials(cellID); ++material ) {
        auto ID    = getMaterialIDNotSafe(cellID, material);
        if( ID == MonteRay_MaterialSpec::NULL_MATERIAL ) continue;

        auto density = getMaterialDensity(cellID, material);

        sum += func( ID, density, temp );
    }
    return sum;
}

template<typename FUNC_T, typename SomeParticle_t, typename T>
T
MonteRay_MaterialProperties::getXsecSum( FUNC_T& func, const SomeParticle_t& p) const {
    Cell_Index_t cellID = getCellIndex(p);
    return getFuncSumByCell( func, cellID);
}

template<typename FUNC_T, typename SomeParticle_t, typename T>
T
MonteRay_MaterialProperties::getXsecSumPerMass( FUNC_T& func, const SomeParticle_t& p) const {
    Cell_Index_t cellID = getCellIndex(p);
    Temperature_t temp = getTemperature(cellID);

    T sum = T(0);
    for( Material_Index_t material=0; material < getNumMaterials(cellID); ++material ) {
        auto ID    = getMaterialIDNotSafe(cellID, material);
        if( ID == MonteRay_MaterialSpec::NULL_MATERIAL ) continue;

        auto density = getMaterialDensity(cellID, material);

        sum += func( ID, density, temp ) / density;
    }
    return sum;
}

} // end namespace


#endif /* MATERIALPROPERTIES_HH_ */
