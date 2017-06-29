#ifndef MONTERAY_SETUPMATERIALPROPERTIES_HH_
#define MONTERAY_SETUPMATERIALPROPERTIES_HH_

#include "MonteRay_MaterialProperties.hh"
#include <vector>

namespace MonteRay{

///\brief Eases the setup and initialization of the MaterialProperties class

///\details
template<typename T>
class MonteRay_SetupMaterialProperties : public MonteRay_MaterialProperties {
public:
    typedef MonteRay_MaterialProperties Base_t;
    typedef MonteRay_MaterialSpec MaterialSpec;

    ///Default Ctor
    MonteRay_SetupMaterialProperties(void) {}
    ///Ctor initialized by total number of cells.
    MonteRay_SetupMaterialProperties(const unsigned int&);
    ///Ctor initialized by matIDs, density, and total number of cells (indexes).
    MonteRay_SetupMaterialProperties(const std::vector<int>&, const std::vector<T>&, const std::size_t);

    // Useful testing functions. Move out to own derived class.
    void setSingleMaterial( const MaterialSpec&, const size_t);
    void setSingleMaterial( const MaterialSpec&, const size_t, const double temp);
    void setSingleTemperature(T, const size_t);
    void editCellMaterial( const MaterialSpec&, const unsigned& );
    void addCellMaterial( const MaterialSpec&, const unsigned& );
    void removeCellMaterial( const MaterialSpec&, const unsigned& );
    void replaceCellMaterial( const MaterialSpec&, const MaterialSpec&, const unsigned& );

    template<typename rangeType>
    void extract( const MonteRay_SetupMaterialProperties<T>& A, const rangeType& obj);

    using Base_t::setMaterialDescription;
    using Base_t::initializeMaterialDescription;
    using Base_t::size;

};

template <typename T>
MonteRay_SetupMaterialProperties<T>::MonteRay_SetupMaterialProperties(const unsigned int& ntotalcells ) :
	MonteRay_MaterialProperties( ntotalcells ) {}

template <typename T>
MonteRay_SetupMaterialProperties<T>::MonteRay_SetupMaterialProperties( const std::vector<int>& matid,
                                                     const std::vector<T>& dens,
                                                     const size_t ntotalcells ) :
    MonteRay_MaterialProperties( matid, dens, ntotalcells ) {}

template<typename T>
void
MonteRay_SetupMaterialProperties<T>::setSingleMaterial( const MaterialSpec& matSpec, const size_t NTotalCells )
{
    std::vector<int>    materialVec( NTotalCells, matSpec.getID() );
    std::vector<double> densityVec ( NTotalCells, matSpec.getDensity() );

    initializeMaterialDescription( materialVec, densityVec, NTotalCells );
}

template<typename T>
void
MonteRay_SetupMaterialProperties<T>::setSingleMaterial( const MaterialSpec& matSpec, const size_t NTotalCells, const double temp )
{
    std::vector<int>    materialVec( NTotalCells, matSpec.getID() );
    std::vector<double> densityVec ( NTotalCells, matSpec.getDensity() );
    std::vector<double> tempVec ( NTotalCells, temp );

    initializeMaterialDescription( materialVec, densityVec, tempVec, NTotalCells );
}

template<typename T>
void
MonteRay_SetupMaterialProperties<T>::setSingleTemperature(T temperature, const size_t NTotalCells) {
	std::vector<T> temperatures(NTotalCells, temperature);
	setCellTemperatures(temperatures);
}


template<typename T>
void
MonteRay_SetupMaterialProperties<T>::addCellMaterial( const MaterialSpec& matSpec, const unsigned& cellID )
{
    Base_t::addCellMaterial(cellID, matSpec.getID(), matSpec.getDensity() );
}

template<typename T>
void
MonteRay_SetupMaterialProperties<T>::removeCellMaterial( const MaterialSpec& matSpec, const unsigned& cellID )
{
    if( size() == 0 ) {
        throw std::runtime_error( "Unable to re-specify material.  Material description container is empty.");
    }

    removeMaterial(cellID, matSpec.getID());
}

template<typename T>
void
MonteRay_SetupMaterialProperties<T>::replaceCellMaterial( const MaterialSpec& remove, const MaterialSpec& replacement, const unsigned& cellID )
{
    removeCellMaterial( remove, cellID );
    addCellMaterial( replacement, cellID );
}

template<typename T>
template<typename rangeType>
void
MonteRay_SetupMaterialProperties<T>::extract( const MonteRay_SetupMaterialProperties<T>& A, const rangeType& obj) {
    clear();
    for(typename rangeType::const_iterator i=obj.begin(); i != obj.end(); ++i) {
        add( A.getCell(*i) );
    }
}

} // end namespace

#endif /* MONTERAY_MATERIALPROPERTIES_HH_ */
