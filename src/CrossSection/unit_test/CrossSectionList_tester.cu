#include <UnitTest++.h>

#include "CrossSectionList.hh"
#include "GPUUtilityFunctions.hh"
#include <sstream>
#include <memory>

namespace CrossSectionList_tester_namespace {

using namespace MonteRay;

SUITE( CrossSectionList_tester ) {


    TEST( CrossSectionList_CPU ) {
        std::vector<double> energies = {0, 1, 2, 3};
        std::vector<double> xsecs = {4, 3, 2, 1};
        int ZAID = 1001;

        CrossSectionList::Builder xsListBuilder;
        xsListBuilder.add(  CrossSectionBuilder( ZAID, energies, xsecs ).construct() );
        auto xsList = xsListBuilder.build();
        CHECK_EQUAL( 4, xsList.getXS(0).size() );
        CHECK_EQUAL( 0, xsList.getXS(0).getID() );
        CHECK_EQUAL( 1, xsList.size() );
     }

    CUDA_CALLABLE_KERNEL kernelGetSize( const CrossSection* xs, int* value) {
        int size = xs->size();
        value[0] = size;
    }

    TEST( CrossSection_from_CrossSectionList_GPU ) {
        std::vector<double> energies = {0, 1, 2, 3};
        std::vector<double> xsecs = {4, 3, 2, 1};
        int ZAID = 1001;

        CrossSectionList::Builder xsListBuilder;
        xsListBuilder.add(  CrossSectionBuilder( ZAID, energies, xsecs ).construct() );

        auto xsList = xsListBuilder.build();
        managed_vector<int> value;
        value.push_back( -1 );

#ifdef __CUDACC__
        kernelGetSize<<<1,1>>>( &(xsList.getXS(0)), value.data() );
#else
        kernelGetSize( &(xsList.getXS(0)), value.data() );
#endif
        deviceSynchronize();
        CHECK_EQUAL( 4, value[0] );
     }

    TEST( CrossSectionList_CPU_add_same_XS_twice ) {
        std::vector<double> energies = {0, 1, 2, 3};
        std::vector<double> xsecs = {4, 3, 2, 1};
        int ZAID = 1001;

        CrossSectionList::Builder xsListBuilder;
        xsListBuilder.add(  CrossSectionBuilder( ZAID, energies, xsecs ).construct() );
        xsListBuilder.add(  CrossSectionBuilder( ZAID, energies, xsecs ).construct() );

        auto xsList = xsListBuilder.build();
        CHECK_EQUAL( 1, xsList.size() );
    }

    TEST( CrossSectionList_CPU_add_two_XSs ) {
        CrossSectionList::Builder xsListBuilder;

        std::vector<double> energies1 = {0, 1, 2, 3};
        std::vector<double> xsecs1 = {4, 3, 2, 1};
        int ZAID = 1001;
        xsListBuilder.add(  CrossSectionBuilder( ZAID, energies1, xsecs1 ).construct() );

        ZAID = 1002;
        std::vector<double> energies2 = {0, 1, 2, 3, 4};
        std::vector<double> xsecs2 = {4, 3, 2, 1, 0.5};
        xsListBuilder.add(  CrossSectionBuilder( ZAID, energies2, xsecs2 ).construct() );

        auto xsList = xsListBuilder.build();
        CHECK_EQUAL( 2, xsList.size() );
        CHECK_EQUAL( 0, xsList.getXS(0).getID() );
        CHECK_EQUAL( 4, xsList.getXS(0).size() );
        CHECK_EQUAL( 1, xsList.getXS(1).getID() );
        CHECK_EQUAL( 5, xsList.getXS(1).size() );
    }

    TEST( read_write_CrossSectionList ){
      std::stringstream stream;

      CrossSectionList::Builder xsListBuilder;

      std::vector<double> energies1 = {0, 1, 2, 3};
      std::vector<double> xsecs1 = {4, 3, 2, 1};
      int ZAID = 1001;
      xsListBuilder.add(  CrossSectionBuilder( ZAID, energies1, xsecs1 ).construct() );

      ZAID = 1002;
      std::vector<double> energies2 = {0, 1, 2, 3, 4};
      std::vector<double> xsecs2 = {4, 3, 2, 1, 0.5};
      xsListBuilder.add(  CrossSectionBuilder( ZAID, energies2, xsecs2 ).construct() );
      auto xsList = xsListBuilder.build();

      xsList.write(stream);
      xsListBuilder.read(stream);
      auto newXSList = xsListBuilder.build();

      CHECK_EQUAL( 2, newXSList.size() );
      CHECK_EQUAL( 0, newXSList.getXS(0).getID() );
      CHECK_EQUAL( 4, newXSList.getXS(0).size() );
      CHECK_EQUAL( 1, newXSList.getXS(1).getID() );
      CHECK_EQUAL( 5, newXSList.getXS(1).size() );
    }
}

} // end namespace
