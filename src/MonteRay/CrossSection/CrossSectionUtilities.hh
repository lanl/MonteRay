#ifndef CROSSSECTIONUTILITIES_HH_
#define CROSSSECTIONUTILITIES_HH_

#include <vector>
#include <list>
#include <utility>
#include <cstddef>
#include <functional>
#include <iostream>

namespace MonteRay {

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

template <typename T>
typename std::enable_if< has_member_func_totalXS_with_energy_temp_and_index<T>::value, double>::type
getTotal(const T& CrossSection, size_t index) {
    return CrossSection.TotalXsec(index);
}

typedef std::function<double (size_t index)  > xsByIndexFunct_t;
typedef std::function<double (double E) > toEnergyFunc_t;
typedef std::function<double (double E)  > totalXSFunct_t;

typedef std::vector<std::pair<double,double>> xsGrid_t;
typedef std::list<std::pair<double,double>> linearGrid_t;

template<typename CROSS_SECTION_T, typename CONTAINER_T>
CONTAINER_T
createXSGrid(const CROSS_SECTION_T& CrossSection, const toEnergyFunc_t& toEnergyFunc, const xsByIndexFunct_t& xsByIndexFunc)
{
    CONTAINER_T linearGrid;

    // build initial grid;
    for( unsigned i=0; i<CrossSection.getEnergyGrid().GridSize(); ++i ){
        double energy = toEnergyFunc( (CrossSection.getEnergyGrid())[i] );
        double totalXS = xsByIndexFunc(i);
        linearGrid.push_back( std::make_pair(energy, totalXS ) );
    }
    return linearGrid;
}

void
thinGrid(const totalXSFunct_t& xsFunc, linearGrid_t& linearGrid, double max_error = 0.1);

void
addPointsToGrid(const totalXSFunct_t& xsFunc, linearGrid_t& linearGrid, double max_error = 0.1);

bool
checkGrid(const totalXSFunct_t& xsFunc, linearGrid_t& linearGrid, double max_error=0.1, unsigned nIntermediateBins=1000);

}
#endif /* CROSSSECTIONUTILITIES_HH_ */
