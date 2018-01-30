#ifndef MONTERAYCROSSSECTION_HH_
#define MONTERAYCROSSSECTION_HH_

#include <iostream>
#include <vector>
#include <stdexcept>
#include <list>
#include <functional>
#include <type_traits>
#include <cmath>

#include "MonteRayDefinitions.hh"
#include "HashLookup.h"
#include "MonteRayConstants.hh"

namespace MonteRay{

struct MonteRayCrossSection {
	int id;
    unsigned numPoints;
    gpuFloatType_t AWR;
    gpuFloatType_t* energies;
    gpuFloatType_t* totalXS;
    ParticleType_t ParticleType;
};

void ctor(struct MonteRayCrossSection*, unsigned num);
void dtor(struct MonteRayCrossSection*);
void copy(struct MonteRayCrossSection* pCopy, struct MonteRayCrossSection* pOrig );

void cudaDtor(MonteRayCrossSection*);
void cudaCtor(MonteRayCrossSection*, unsigned num);
void cudaCtor(MonteRayCrossSection*, MonteRayCrossSection*);

CUDA_CALLABLE_MEMBER
gpuFloatType_t getEnergy(const struct MonteRayCrossSection* pXS, unsigned i );

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXSByIndex(const struct MonteRayCrossSection* pXS, unsigned i );

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXS(const struct MonteRayCrossSection* pXS, gpuFloatType_t E );

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXS(const struct MonteRayCrossSection* pXS, const struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E );

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXSByIndex(const struct MonteRayCrossSection* pXS, unsigned i, gpuFloatType_t E );

CUDA_CALLABLE_MEMBER
unsigned getIndex(const struct MonteRayCrossSection* pXS, gpuFloatType_t E );

CUDA_CALLABLE_MEMBER
unsigned getIndex(const struct MonteRayCrossSection* pXS, const struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E );

CUDA_CALLABLE_MEMBER
unsigned getIndexBinary(const struct MonteRayCrossSection* pXS, unsigned lower, unsigned upper, gpuFloatType_t value );

CUDA_CALLABLE_MEMBER
unsigned getIndexLinear(const struct MonteRayCrossSection* pXS, unsigned lower, unsigned upper, gpuFloatType_t value );

CUDA_CALLABLE_MEMBER
gpuFloatType_t getAWR(const struct MonteRayCrossSection* pXS);

CUDA_CALLABLE_MEMBER
ParticleType_t getParticleType(const struct MonteRayCrossSection* pXS);

CUDA_CALLABLE_MEMBER
void setParticleType( struct MonteRayCrossSection* pXS, ParticleType_t type);

CUDA_CALLABLE_MEMBER
int getID(const struct MonteRayCrossSection* pXS);

CUDA_CALLABLE_MEMBER
void setID(struct MonteRayCrossSection* pXS, unsigned i);

CUDA_CALLABLE_KERNEL void kernelGetTotalXS(const struct MonteRayCrossSection* pXS, const HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t* result);

CUDA_CALLABLE_KERNEL void kernelGetTotalXS(const struct MonteRayCrossSection* pXS, gpuFloatType_t E, gpuFloatType_t* result);

//class ContinuousNeutron;

class MonteRayCrossSectionHost {
public:
    MonteRayCrossSectionHost(unsigned num);
    ~MonteRayCrossSectionHost();

    void copyToGPU(void);

    int getID(void) const { return MonteRay::getID( xs ); }
    void setID(unsigned id) { MonteRay::setID( xs, id ); }
    unsigned size(void) const { return xs->numPoints; }
    ParticleType_t getParticleType(void) const { return xs->ParticleType; }
    void setParticleType(ParticleType_t type) { xs->ParticleType = type; }
    gpuFloatType_t getEnergy(unsigned i) const { return MonteRay::getEnergy(xs, i); }
    gpuFloatType_t getTotalXSByIndex(unsigned i) const { return MonteRay::getTotalXSByIndex(xs, i); }
    gpuFloatType_t getTotalXSByHashIndex(const struct HashLookup* pHash, unsigned i, gpuFloatType_t E) const;

    gpuFloatType_t getTotalXS( gpuFloatType_t E ) const { return MonteRay::getTotalXS(xs, E); }
    gpuFloatType_t getTotalXS( const struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E ) const;

    void setEnergy(unsigned i, gpuFloatType_t e) { xs->energies[i] = e; }
    unsigned getIndex( gpuFloatType_t e ) const { return MonteRay::getIndex( xs, e); }
    unsigned getIndex( const HashLookupHost* pHost, unsigned hashBin, gpuFloatType_t e ) const;

    void setTotalXS(unsigned i, gpuFloatType_t value) { xs->totalXS[i] = value; }

    void setTotalXS(unsigned i, gpuFloatType_t E, gpuFloatType_t value) {
    	if( i >= size() ) {
    		throw std::runtime_error( "Error: MonteRayCrossSectionHost::setTotalXS, invalid index");
    	}
        xs->energies[i] = E;
        xs->totalXS[i] = value;
    }

    gpuFloatType_t getAWR(void) const {return MonteRay::getAWR(xs); }
    void setAWR(gpuFloatType_t value) { xs->AWR = value; }

    void write(std::ostream& outfile) const;
    void  read(std::istream& infile);

    void write( const std::string& filename );
    void read( const std::string& filename );

    struct MonteRayCrossSection* getXSPtr(void) { return xs;}
    struct MonteRayCrossSection& getXSRef(void) { return *xs;}

    void load(struct MonteRayCrossSection* ptrXS );

    typedef std::vector<std::pair<double,double>> xsGrid_t;
    typedef std::list<std::pair<double,double>> linearGrid_t;
    typedef std::function<double (double E)  > totalXSFunct_t;
    typedef std::function<double (double E, size_t index)  > xsByIndexFunct_t;
    typedef std::function<double (double E) > toEnergyFunc_t;
    static constexpr double maxError = 0.1;
    static const unsigned nBinsToCheck = 1000;

    template <typename T>
    struct has_member_func_totalXS_with_energy_temp_and_index {
    	template <typename C>
    	static auto test(double x) -> decltype( std::declval<C>().TotalXsec(x, -1.0, 0), std::true_type() );

    	template <typename>
    	static std::false_type test( ... );

    	typedef decltype( test<T>(1.0) ) CheckType;
    	static const bool value = std::is_same<std::true_type,CheckType>::value;
    };

    template <typename T>
    typename std::enable_if< !has_member_func_totalXS_with_energy_temp_and_index<T>::value, double>::type
    getTotal(const T& CrossSection, double& E) {
    	return CrossSection.TotalXsec(E);
    }

    template<typename T>
    typename std::enable_if< !has_member_func_totalXS_with_energy_temp_and_index<T>::value, double>::type
    getTotal(const T& CrossSection, double E, size_t index) {
    	return CrossSection.TotalXsec(E, index);
    }

    template <typename T>
    typename std::enable_if< has_member_func_totalXS_with_energy_temp_and_index<T>::value, double>::type
    getTotal(const T& CrossSection, double E ) {
    	return CrossSection.TotalXsec(E, -1.0);
    }

    template <typename T>
    typename std::enable_if< has_member_func_totalXS_with_energy_temp_and_index<T>::value, double>::type
    getTotal(const T& CrossSection, double E, size_t index) {
    	return CrossSection.TotalXsec(E, -1.0, index);
    }

    template<typename CROSS_SECTION_T, typename GRID_T>
    GRID_T
    createInitialGrid(const CROSS_SECTION_T& CrossSection, const toEnergyFunc_t& toEnergyFunc, const xsByIndexFunct_t& xsByIndexFunc ) const{
    	GRID_T linearGrid;

    	// build initial grid;
    	for( unsigned i=0; i<CrossSection.getEnergyGrid().GridSize(); ++i ){
    		double energy = toEnergyFunc( (CrossSection.getEnergyGrid())[i] );
    		double totalXS = xsByIndexFunc( energy, i);
    		linearGrid.push_back( std::make_pair(energy, totalXS ) );
    	}
    	return linearGrid;
    }

    static void thinGrid(const totalXSFunct_t& xsFunc, linearGrid_t& linearGrid, double max_error);
    void addPointsToGrid(const totalXSFunct_t& xsFunc, linearGrid_t& grid, double max_error) const;
    bool checkGrid(const totalXSFunct_t& xsFunc, linearGrid_t& grid, double max_error, unsigned nIntermediateBins) const;

    template<typename T>
    void load(const T& CrossSection, double error = maxError, unsigned nBinsToVerify = nBinsToCheck ) {
    	typedef std::vector<double> xsec_t;
    	xsGrid_t xsGrid;

    	toEnergyFunc_t toEnergyFunc;
    	ParticleType_t ParticleType;
    	if( CrossSection.getType() == "neutron" ) {
    		ParticleType = neutron;
    		toEnergyFunc = [&] (double E) -> double {
    			return E;
    		};
    		auto xsFuncByIndex = [&] ( double E, size_t index ) {
    			return getTotal<T>( CrossSection, E, index );
    		};
    		xsGrid = createInitialGrid<T,xsGrid_t>(CrossSection, toEnergyFunc, xsFuncByIndex);
    	}
    	if( CrossSection.getType() == "photon" ) {
    		ParticleType = photon;
    		toEnergyFunc = [&] (double E) -> double {
    			return std::exp(E);
    		};
    		auto xsFunc = [&] ( double E ) {
    			return getTotal<T>( CrossSection, E );
    		};
    		auto xsFuncByIndex = [&] ( double E, size_t index ) {
    			return getTotal<T>( CrossSection, E, index );
    		};

    		linearGrid_t linearGrid = createInitialGrid<T,linearGrid_t>( CrossSection, toEnergyFunc, xsFuncByIndex );
    		thinGrid( xsFunc, linearGrid, error );

    		bool done = true;
    		do {
    			addPointsToGrid( xsFunc, linearGrid, error );
    			done = checkGrid( xsFunc, linearGrid, error, nBinsToVerify );
    		} while (!done);

    		xsGrid.reserve( linearGrid.size() );
    		std::copy( std::begin(linearGrid), std::end(linearGrid), std::back_inserter(xsGrid));
    	}

    	unsigned num = xsGrid.size();
    	dtor( xs );
    	ctor( xs, num );
    	setAWR( CrossSection.getAWR() );
    	xs->ParticleType = ParticleType;

    	for( unsigned i=0; i<num; ++i ){
    		xs->energies[i] = xsGrid[i].first;
    		xs->totalXS[i] = xsGrid[i].second;
    	}
    }

    const MonteRayCrossSection* getPtr() const { return xs; }

private:
    struct MonteRayCrossSection* xs;
    MonteRayCrossSection* temp;
    bool cudaCopyMade;

public:
    MonteRayCrossSection* xs_device;

};

gpuFloatType_t launchGetTotalXS( MonteRayCrossSectionHost* pXS, gpuFloatType_t energy);


}


#endif /* MONTERAYCROSSSECTION_HH_ */
