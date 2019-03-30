#include <UnitTest++.h>

#include "CrossSectionList.hh"
#include "GPUUtilityFunctions.hh"
#include <memory>

namespace CrossSectionList_tester_namespace {

using namespace MonteRay;

SUITE( CrossSectionList_tester ) {


    TEST( CrossSectionList_CPU ) {
        std::vector<double> energies = {0, 1, 2, 3};
        std::vector<double> xsecs = {4, 3, 2, 1};
        int ZAID = 1001;

        CrossSectionList xsList;
        xsList.add(  CrossSectionBuilder( ZAID, energies, xsecs ).construct() );
        CHECK_EQUAL( 4, xsList.getXS(0).size() );
        CHECK_EQUAL( 0, xsList.getXS(0).getID() );
        CHECK_EQUAL( 1, xsList.size() );
     }

    CUDA_CALLABLE_KERNEL kernelGetSize( CrossSection* xs, int* value) {
        int size = xs->size();
        value[0] = size;
    }

    TEST( CrossSection_from_CrossSectionList_GPU ) {
        std::vector<double> energies = {0, 1, 2, 3};
        std::vector<double> xsecs = {4, 3, 2, 1};
        int ZAID = 1001;

        CrossSectionList xsList;
        xsList.add(  CrossSectionBuilder( ZAID, energies, xsecs ).construct() );

        managed_vector<int> value;
        value.push_back( -1 );

#ifdef __CUDACC__
        kernelGetSize<<<1,1>>>( xsList.getXSPtr(0), value.data() );
#else
        kernelGetSize( xsList.getXSPtr(0), value.data() );
#endif
        deviceSynchronize();
        CHECK_EQUAL( 4, value[0] );
     }

    TEST( CrossSectionList_CPU_getXSByZAID ) {
         std::vector<double> energies = {0, 1, 2, 3};
         std::vector<double> xsecs = {4, 3, 2, 1};
         int ZAID = 1001;

         CrossSectionList xsList;
         xsList.add(  CrossSectionBuilder( ZAID, energies, xsecs ).construct() );

         CrossSection* pXS;
         pXS = xsList.getXSByZAID( 1001 );
         CHECK_EQUAL( 4, pXS->size() );
      }

    TEST( CrossSectionList_CPU_add_same_XS_twice ) {
        std::vector<double> energies = {0, 1, 2, 3};
        std::vector<double> xsecs = {4, 3, 2, 1};
        int ZAID = 1001;

        CrossSectionList xsList;
        xsList.add(  CrossSectionBuilder( ZAID, energies, xsecs ).construct() );
        xsList.add(  CrossSectionBuilder( ZAID, energies, xsecs ).construct() );

        CHECK_EQUAL( 1, xsList.size() );
    }

    TEST( CrossSectionList_CPU_add_two_XSs ) {
        std::vector<double> energies1 = {0, 1, 2, 3};
        std::vector<double> xsecs1 = {4, 3, 2, 1};
        int ZAID = 1001;

        CrossSectionList xsList;
        xsList.add(  CrossSectionBuilder( ZAID, energies1, xsecs1 ).construct() );

        ZAID = 1002;
        std::vector<double> energies2 = {0, 1, 2, 3, 4};
        std::vector<double> xsecs2 = {4, 3, 2, 1, 0.5};

        xsList.add(  CrossSectionBuilder( ZAID, energies2, xsecs2 ).construct() );

        CHECK_EQUAL( 2, xsList.size() );
        CHECK_EQUAL( 0, xsList.getXS(0).getID() );
        CHECK_EQUAL( 4, xsList.getXS(0).size() );
        CHECK_EQUAL( 1, xsList.getXS(1).getID() );
        CHECK_EQUAL( 5, xsList.getXS(1).size() );
    }

    TEST( CrossSection_AWR ) {
        std::vector<double> energies1 = {0, 1, 2, 3};
        std::vector<double> xsecs1 = {4, 3, 2, 1};
        int ZAID = 1001;

        CrossSectionList xsList;
        CrossSectionBuilder xsbuilder( ZAID, energies1, xsecs1 );
        xsbuilder.setAWR( 15.0 );

        CrossSection xs = xsbuilder.construct();
        CHECK_CLOSE( 15.0, xs.getAWR(), 1e-6);
    }

    TEST( CrossSection_default_particle_type ) {
        std::vector<double> energies1 = {0, 1, 2, 3};
        std::vector<double> xsecs1 = {4, 3, 2, 1};
        int ZAID = 1001;

        CrossSectionList xsList;
        CrossSectionBuilder xsbuilder( ZAID, energies1, xsecs1 );

        CrossSection xs = xsbuilder.construct();
        CHECK_EQUAL( neutron, xs.getParticleType() );
    }


}

} // end namespace
