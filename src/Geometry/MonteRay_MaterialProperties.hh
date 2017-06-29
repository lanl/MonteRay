#ifndef MONTERAY_MATERIALPROPERTIES_HH_
#define MONTERAY_MATERIALPROPERTIES_HH_

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <ostream>
#include <fstream>
#include <memory>

#include "MonteRay_CellProperties.hh"
#include "MonteRay_MaterialProperties_FlatLayout.hh"

#include "MonteRayDefinitions.hh"

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

#ifdef CUDA
void cudaCtor(struct MonteRay_MaterialProperties_Data*,struct MonteRay_MaterialProperties_Data*);
void cudaDtor(struct MonteRay_MaterialProperties_Data*);
#endif

#ifdef CUDA
__device__ __host__
#endif
size_t getNumCells(struct MonteRay_MaterialProperties_Data* ptr );

#ifdef CUDA
__device__ __host__
#endif
MonteRay_MaterialProperties_Data::offset_t getNumMats(struct MonteRay_MaterialProperties_Data* ptr, unsigned i );

#ifdef CUDA
__device__ __host__
#endif
MonteRay_MaterialProperties_Data::Density_t getDensity(struct MonteRay_MaterialProperties_Data* ptr, unsigned cellNum, unsigned matNum );

#ifdef CUDA
__device__ __host__
#endif
MonteRay_MaterialProperties_Data::MatID_t getMatID(struct MonteRay_MaterialProperties_Data* ptr, unsigned cellNum, unsigned matNum );

#ifdef CUDA
__global__ void kernelGetNumCells(MonteRay_MaterialProperties_Data* mp, unsigned* results );
#endif

#ifdef CUDA
__global__ void kernelGetNumMaterials(MonteRay_MaterialProperties_Data* mp, unsigned cellNum, MonteRay_MaterialProperties_Data::Material_Index_t* results );
#endif

#ifdef CUDA
__global__ void kernelGetMaterialID(MonteRay_MaterialProperties_Data* mp, unsigned cellNum, unsigned i, MonteRay_MaterialProperties_Data::MatID_t* results );
#endif

#ifdef CUDA
__global__ void kernelGetMaterialDensity(MonteRay_MaterialProperties_Data* mp, unsigned cellNum, unsigned i, MonteRay_MaterialProperties_Data::Density_t* results );
#endif

#ifdef CUDA
__global__ void kernelSumMatDensity(MonteRay_MaterialProperties_Data* mp, MonteRay_MaterialProperties_Data::MatID_t matIndex, MonteRay_MaterialProperties_Data::Density_t* results );
#endif


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
        disableReduction();
    }

    ///Ctor initialized by total number of cells.
    MonteRay_MaterialProperties(const std::size_t& nCells) {
        pMemoryLayout.reset( new MemoryLayout_t(nCells) );
        disableReduction();
    }

    ///Ctor initialized by matIDs, density, and total number of cells (indexes).
    template<typename MaterialIDType, typename DensityType>
    MonteRay_MaterialProperties(const std::vector<MaterialIDType>& IDs, const std::vector<DensityType>& dens, const std::size_t nCells){
        pMemoryLayout.reset( new MemoryLayout_t(IDs, dens, nCells) );
    }

    ///Default Dtor
    virtual ~MonteRay_MaterialProperties(void) {
#ifdef CUDA
    	cudaDtor();
#endif
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
    }

    template<typename MaterialIDType, typename DensityType, typename TempType>
    void initializeMaterialDescription( const std::vector<MaterialIDType>& IDs, const std::vector<DensityType>& dens, const std::vector<TempType>& temps, const std::size_t nCells) {
        pMemoryLayout->template initializeMaterialDescription<MaterialIDType,DensityType,TempType>( IDs, dens, temps, nCells );
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

    offset_t* getOffsetData(void) { return pMemoryLayout->getOffsetData(); }
    MatID_t* getMaterialIDData(void) { return pMemoryLayout->getMaterialIDData(); }
    Density_t* getMaterialDensityData(void) { return pMemoryLayout->getMaterialDensityData(); }

    void disableReduction() { pMemoryLayout->disableReduction(); }

    Density_t launchSumMatDensity(MatID_t matIndex) const;
    Density_t sumMatDensity( MatID_t matIndex) const;

    size_t launchGetNumCells(void) const;
    Material_Index_t launchGetNumMaterials( Cell_Index_t cellID ) const;
    MatID_t launchGetMaterialID( Cell_Index_t cellID, Material_Index_t i ) const;
    Density_t launchGetMaterialDensity( Cell_Index_t cellID, Material_Index_t i ) const;

private:
    pMemoryLayout_t pMemoryLayout;
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
