#include <UnitTest++.h>

#include <fstream>

#include "HashLookup.hh"
#include "MonteRayCrossSection.hh"
#include "HashLookup_test_helper.hh"

SUITE( HashLookup_tester ) {
    using namespace MonteRay;
    TEST( setup ) {
        //gpuCheck();
    }
    TEST( ctor ) {
        HashLookupHost hash(10);
        CHECK_EQUAL(10, hash.getMaxNumIsotopes() );
        CHECK_EQUAL(0, hash.getNumIsotopes() );
        CHECK_EQUAL(8192, hash.getNBins() );
    }
    TEST( add_an_isotope_checkMinAndMax ){
        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(4);
        xs->setTotalXS(0, 0.1, 4.0 );
        xs->setTotalXS(1, 1.0, 3.0 );
        xs->setTotalXS(2, 2.0, 2.0 );
        xs->setTotalXS(3, 3.0, 1.0 );

        HashLookupHost hash(10);
        hash.addIsotope( xs );
        CHECK_EQUAL(1, hash.getNumIsotopes() );
        CHECK_CLOSE(0.1, hash.getMinEnergy(), 1e-5);
        CHECK_CLOSE(3.0, hash.getMaxEnergy(), 1e-5);
        delete xs;
    }

    TEST( add_two_isotopes_checkMinAndMax ){
        MonteRayCrossSectionHost* xs1 = new MonteRayCrossSectionHost(4);
        xs1->setTotalXS(0, 0.5, 4.0 );
        xs1->setTotalXS(1, 1.0, 3.0 );
        xs1->setTotalXS(2, 2.0, 2.0 );
        xs1->setTotalXS(3, 3.0, 1.0 );

        MonteRayCrossSectionHost* xs2 = new MonteRayCrossSectionHost(4);
        xs2->setTotalXS(0, 0.25, 4.0 );
        xs2->setTotalXS(1, 1.0, 3.0 );
        xs2->setTotalXS(2, 2.0, 2.0 );
        xs2->setTotalXS(3, 3.5, 1.0 );

        HashLookupHost hash(10);
        hash.addIsotope( xs1 );
        CHECK_EQUAL(1, hash.getNumIsotopes() );
        CHECK_CLOSE(0.5, hash.getMinEnergy(), 1e-5);
        CHECK_CLOSE(3.0, hash.getMaxEnergy(), 1e-5);

        hash.addIsotope( xs2 );
        CHECK_EQUAL(2, hash.getNumIsotopes() );
        CHECK_CLOSE(0.25, hash.getMinEnergy(), 1e-5);
        CHECK_CLOSE(3.5, hash.getMaxEnergy(), 1e-5);

        delete xs1;
        delete xs2;
    }
    TEST( add_thre_isotopes_checkMinAndMax ){
        MonteRayCrossSectionHost* xs1 = new MonteRayCrossSectionHost(4);
        xs1->setTotalXS(0, 0.5, 4.0 );
        xs1->setTotalXS(1, 1.0, 3.0 );
        xs1->setTotalXS(2, 2.0, 2.0 );
        xs1->setTotalXS(3, 3.0, 1.0 );

        MonteRayCrossSectionHost* xs2 = new MonteRayCrossSectionHost(4);
        xs2->setTotalXS(0, 0.25, 4.0 );
        xs2->setTotalXS(1, 1.0, 3.0 );
        xs2->setTotalXS(2, 2.0, 2.0 );
        xs2->setTotalXS(3, 3.5, 1.0 );

        MonteRayCrossSectionHost* xs3 = new MonteRayCrossSectionHost(4);
        xs3->setTotalXS(0, 0.75, 4.0 );
        xs3->setTotalXS(1, 1.0, 3.0 );
        xs3->setTotalXS(2, 2.0, 2.0 );
        xs3->setTotalXS(3, 3.25, 1.0 );

        HashLookupHost hash(10);
        hash.addIsotope( xs1 );
        hash.addIsotope( xs2 );
        hash.addIsotope( xs3 );
        CHECK_EQUAL(3, hash.getNumIsotopes() );
        CHECK_CLOSE(0.25, hash.getMinEnergy(), 1e-5);
        CHECK_CLOSE(3.5, hash.getMaxEnergy(), 1e-5);
        delete xs1;
        delete xs2;
        delete xs3;
    }
    TEST( get_hash_bin_index_by_energy ){
        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(2);
        xs->setTotalXS(0, 1.0, 4.0 );
        xs->setTotalXS(1, 10.0, 3.0 );

        HashLookupHost hash(1, 10);
        hash.addIsotope( xs );
        CHECK_EQUAL(1, hash.getNumIsotopes() );
        CHECK_EQUAL(10, hash.getNBins() );
        CHECK_CLOSE(1.0, hash.getMinEnergy(), 1e-5);
        CHECK_CLOSE(10.0, hash.getMaxEnergy(), 1e-5);
        CHECK_EQUAL( 3, hash.getHashBin( 2.0 ) );
        CHECK_EQUAL( 0, hash.getHashBin( 0.0 ) );
        CHECK_EQUAL( 0, hash.getHashBin( 1.0 ) );
        CHECK_EQUAL( 0, hash.getHashBin( 1.25 ) );
        CHECK_EQUAL( 9, hash.getHashBin( 9.0 ) );
        CHECK_EQUAL( 9, hash.getHashBin( 10.0 ) );
        CHECK_EQUAL( 9, hash.getHashBin( 11.0 ) );
        delete xs;
    }
    TEST( fill_lower_bin_array ){
        //    	std::cout << "Debug: starting fill_lower_bin_array\n";
        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(10);
        xs->setTotalXS(0, 1.0, 4.0 );
        xs->setTotalXS(1, 1.25, 4.0 );
        xs->setTotalXS(2, 2.0, 4.0 );
        xs->setTotalXS(3, 2.5, 4.0 );
        xs->setTotalXS(4, 3.0, 4.0 );
        xs->setTotalXS(5, 4.0, 4.0 );
        xs->setTotalXS(6, 5.0, 4.0 );
        xs->setTotalXS(7, 7.0, 4.0 );
        xs->setTotalXS(8, 9.0, 4.0 );
        xs->setTotalXS(9, 10.0, 3.0 );

        HashLookupHost hash(4, 10);
        //    	std::cout << "Debug:calling add isotope\n";
        hash.addIsotope( xs );
        CHECK_EQUAL(1, hash.getNumIsotopes() );
        CHECK_EQUAL(10, hash.getNBins() );
        CHECK_CLOSE(1.0, hash.getMinEnergy(), 1e-5);
        CHECK_CLOSE(10.0, hash.getMaxEnergy(), 1e-5);

        unsigned isotopeNum = 0;
        // check getBinBoundIndex
        //    	std::cout << "Debug:calling testing indexs \n";
        CHECK_EQUAL(0, hash.getBinBoundIndex(0,0));
        //    	CHECK_EQUAL(1, hash.getBinBoundIndex(1,0));
        //    	CHECK_EQUAL(2, hash.getBinBoundIndex(2,0));
        //    	CHECK_EQUAL(3, hash.getBinBoundIndex(3,0));
        CHECK_EQUAL(4, hash.getBinBoundIndex(0,1));
        //    	CHECK_EQUAL(19, hash.getBinBoundIndex(3,4));


        // lowest bound should always be 0;

        CHECK_EQUAL(0, hash.getLowerBoundbyIndex( isotopeNum, 0 ) );
        CHECK_EQUAL(2, hash.getUpperBoundbyIndex( isotopeNum, 0 ) );

        CHECK_EQUAL(1, hash.getLowerBoundbyIndex( isotopeNum, 1 ) );
        CHECK_EQUAL(2, hash.getUpperBoundbyIndex( isotopeNum, 1 ) );

        CHECK_EQUAL(1, hash.getLowerBoundbyIndex( isotopeNum, 2 ) );
        CHECK_EQUAL(2, hash.getUpperBoundbyIndex( isotopeNum, 2 ) );

        CHECK_EQUAL(1, hash.getLowerBoundbyIndex( isotopeNum, 3 ) );
        CHECK_EQUAL(4, hash.getUpperBoundbyIndex( isotopeNum, 3 ) );

        CHECK_EQUAL(3, hash.getLowerBoundbyIndex( isotopeNum, 4 ) );
        CHECK_EQUAL(5, hash.getUpperBoundbyIndex( isotopeNum, 4 ) );

        CHECK_EQUAL(7, hash.getLowerBoundbyIndex( isotopeNum, 9 ) );
        CHECK_EQUAL(9, hash.getUpperBoundbyIndex( isotopeNum, 9 ) );
        delete xs;
    }

    TEST( getTotalXS_via_hash ){
        //    	std::cout << "Debug: starting getTotalXS_via_hash\n";
        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(10);
        xs->setTotalXS(0, 1.0, 1.0 );
        xs->setTotalXS(1, 1.25, 4.0 );
        xs->setTotalXS(2, 2.0, 3.0 );
        xs->setTotalXS(3, 2.5, 5.0 );
        xs->setTotalXS(4, 3.0, 4.0 );
        xs->setTotalXS(5, 4.0, 4.0 );
        xs->setTotalXS(6, 5.0, 4.0 );
        xs->setTotalXS(7, 7.0, 4.0 );
        xs->setTotalXS(8, 9.0, 4.0 );
        xs->setTotalXS(9, 10.0, 10.0 );
        CHECK_EQUAL( -1, xs->getID());



        HashLookupHost* hash = new HashLookupHost(4, 10);
        hash->addIsotope( xs );
        CHECK_EQUAL( 0, xs->getID());
        CHECK_EQUAL( 1, hash->getNumIsotopes() );

        gpuFloatType_t energy = 2.25;
        CHECK_EQUAL( 2, getIndex(xs->getXSPtr(), energy  ));
        CHECK_EQUAL( 2, getIndexBinary(xs->getXSPtr(), 0, 9, energy  ));
        CHECK_EQUAL( 2, getIndexLinear(xs->getXSPtr(), 0, 9, energy  ));
        CHECK_CLOSE( 4.0, getTotalXS(xs->getXSPtr(), 2, energy ), 1e-5);
        unsigned hashBin = hash->getHashBin( energy );
        CHECK_EQUAL( 3, hashBin );
        CHECK_EQUAL(1, hash->getLowerBoundbyIndex( xs->getID(), hashBin ) );
        CHECK_EQUAL(4, hash->getUpperBoundbyIndex( xs->getID(), hashBin ) );
        CHECK_EQUAL( 2, getIndexBinary(xs->getXSPtr(), 1, 4, energy  ));
        CHECK_EQUAL( 2, getIndexBinary(xs->getXSPtr(), 1, 4, energy  ));
        CHECK_CLOSE( 4.0, getTotalXS(xs->getXSPtr(), hash->getPtr(), hashBin, energy ), 1e-5);

        // lower end
        energy = 0.1;
        CHECK_EQUAL( 0, getIndexBinary(xs->getXSPtr(), 0, 9, energy  ));
        CHECK_EQUAL( 0, getIndexLinear(xs->getXSPtr(), 0, 9, energy  ));
        hashBin = hash->getHashBin( energy );
        CHECK_EQUAL( 0, hashBin );
        CHECK_EQUAL(0, hash->getLowerBoundbyIndex( xs->getID(), hashBin ) );
        CHECK_EQUAL(2, hash->getUpperBoundbyIndex( xs->getID(), hashBin ) );
        CHECK_CLOSE( 1.0, getTotalXS(xs->getXSPtr(), hash->getPtr(), hashBin, energy ), 1e-5);

        // upper end
        energy = 10.1;
        CHECK_EQUAL( 9, getIndexBinary(xs->getXSPtr(), 0, 9, energy  ));
        CHECK_EQUAL( 9, getIndexLinear(xs->getXSPtr(), 0, 9, energy  ));
        hashBin = hash->getHashBin( energy );
        CHECK_EQUAL( 9, hashBin );
        CHECK_EQUAL(7, hash->getLowerBoundbyIndex( xs->getID(), hashBin ) );
        CHECK_EQUAL(9, hash->getUpperBoundbyIndex( xs->getID(), hashBin ) );
        CHECK_CLOSE( 10.0, getTotalXS(xs->getXSPtr(), hash->getPtr(), hashBin, energy ), 1e-5);

        energy = 1.15;
        hashBin = hash->getHashBin( energy );
        CHECK_EQUAL( 0, hashBin );

        //    	CHECK_EQUAL( -1, xs->getIndex( 3.0) );
        //    	CHECK_EQUAL( -1, xs->getIndex( hash, hashBin, 3.0) );
        //    	CHECK_CLOSE(4.0, xs->getTotalXS( 3.0), 1e-5);
        //    	CHECK_CLOSE(4.0, xs->getTotalXS( hash, 3.0), 1e-5);

        delete xs;
        delete hash;
    }

    TEST( fill_lower_bin_array_two_xs ){
        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(10);
        xs->setTotalXS(0, 1.0, 4.0 );
        xs->setTotalXS(1, 1.3, 4.0 );
        xs->setTotalXS(2, 1.9, 4.0 );
        xs->setTotalXS(3, 2.4, 4.0 );
        xs->setTotalXS(4, 3.0, 4.0 );
        xs->setTotalXS(5, 4.0, 4.0 );
        xs->setTotalXS(6, 5.0, 4.0 );
        xs->setTotalXS(7, 7.0, 4.0 );
        xs->setTotalXS(8, 9.0, 4.0 );
        xs->setTotalXS(9, 10.0, 3.0 );

        MonteRayCrossSectionHost* xs2 = new MonteRayCrossSectionHost(10);
        xs2->setTotalXS(0, 1.0, 4.0 );
        xs2->setTotalXS(1, 1.25, 4.0 );
        xs2->setTotalXS(2, 2.0, 4.0 );
        xs2->setTotalXS(3, 2.5, 4.0 );
        xs2->setTotalXS(4, 3.0, 4.0 );
        xs2->setTotalXS(5, 4.0, 4.0 );
        xs2->setTotalXS(6, 5.0, 4.0 );
        xs2->setTotalXS(7, 7.0, 4.0 );
        xs2->setTotalXS(8, 9.0, 4.0 );
        xs2->setTotalXS(9, 10.0, 3.0 );

        HashLookupHost hash(4, 10);
        hash.addIsotope( xs );
        hash.addIsotope( xs2 );
        CHECK_EQUAL(2, hash.getNumIsotopes() );
        CHECK_EQUAL(10, hash.getNBins() );
        CHECK_CLOSE(1.0, hash.getMinEnergy(), 1e-5);
        CHECK_CLOSE(10.0, hash.getMaxEnergy(), 1e-5);

        unsigned isotopeNum = 1;
        // check getBinBoundIndex
        CHECK_EQUAL(0, hash.getBinBoundIndex(0,0));
        CHECK_EQUAL(1, hash.getBinBoundIndex(1,0));
        //    	CHECK_EQUAL(2, hash.getBinBoundIndex(2,0));
        //    	CHECK_EQUAL(3, hash.getBinBoundIndex(3,0));
        CHECK_EQUAL(4, hash.getBinBoundIndex(0,1));
        //    	CHECK_EQUAL(19, hash.getBinBoundIndex(3,4));


        // lowest bound should always be 0;

        CHECK_EQUAL(0, hash.getLowerBoundbyIndex( isotopeNum, 0 ) );
        CHECK_EQUAL(2, hash.getUpperBoundbyIndex( isotopeNum, 0 ) );

        CHECK_EQUAL(1, hash.getLowerBoundbyIndex( isotopeNum, 1 ) );
        CHECK_EQUAL(2, hash.getUpperBoundbyIndex( isotopeNum, 1 ) );

        CHECK_EQUAL(1, hash.getLowerBoundbyIndex( isotopeNum, 2 ) );
        CHECK_EQUAL(2, hash.getUpperBoundbyIndex( isotopeNum, 2 ) );

        CHECK_EQUAL(1, hash.getLowerBoundbyIndex( isotopeNum, 3 ) );
        CHECK_EQUAL(4, hash.getUpperBoundbyIndex( isotopeNum, 3 ) );

        CHECK_EQUAL(3, hash.getLowerBoundbyIndex( isotopeNum, 4 ) );
        CHECK_EQUAL(5, hash.getUpperBoundbyIndex( isotopeNum, 4 ) );

        CHECK_EQUAL(7, hash.getLowerBoundbyIndex( isotopeNum, 9 ) );
        CHECK_EQUAL(9, hash.getUpperBoundbyIndex( isotopeNum, 9 ) );
        delete xs;
        delete xs2;
    }
    TEST_FIXTURE(HashLookupTestHelper, getLowerBoundOnGPU ){
        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(10);
        xs->setTotalXS(0, 1.0, 4.0 );
        xs->setTotalXS(1, 1.25, 4.0 );
        xs->setTotalXS(2, 2.0, 4.0 );
        xs->setTotalXS(3, 2.5, 4.0 );
        xs->setTotalXS(4, 3.0, 4.0 );
        xs->setTotalXS(5, 4.0, 4.0 );
        xs->setTotalXS(6, 5.0, 4.0 );
        xs->setTotalXS(7, 7.0, 4.0 );
        xs->setTotalXS(8, 9.0, 4.0 );
        xs->setTotalXS(9, 10.0, 3.0 );

        HashLookupHost* hash = new HashLookupHost(4, 10);
        hash->addIsotope( xs );

        xs->copyToGPU();
        hash->copyToGPU();

        unsigned isotopeNum = 0;
        setupTimers();
        unsigned lower = launchGetLowerBoundbyIndex( hash, isotopeNum, 4 );
        CHECK_EQUAL(3, lower );
        stopTimers();

        delete xs;
        delete hash;
    }

    TEST( getTotalXS_via_hash_two_isotopes_different_min_max ){
        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(10);
        xs->setTotalXS(0, 1.0, 1.0 );
        xs->setTotalXS(1, 1.25, 2.0 );
        xs->setTotalXS(2, 2.0, 3.0 );
        xs->setTotalXS(3, 2.5, 4.0 );
        xs->setTotalXS(4, 3.0, 5.0 );
        xs->setTotalXS(5, 4.0, 6.0 );
        xs->setTotalXS(6, 5.0, 7.0 );
        xs->setTotalXS(7, 7.0, 8.0 );
        xs->setTotalXS(8, 9.0, 9.0 );
        xs->setTotalXS(9, 10.0, 10.0 );

        MonteRayCrossSectionHost* xs2 = new MonteRayCrossSectionHost(10);
        xs2->setTotalXS(0, 0.1, 1.0 );
        xs2->setTotalXS(1, 0.125, 2.0 );
        xs2->setTotalXS(2, 0.2, 3.0 );
        xs2->setTotalXS(3, 0.25, 4.0 );
        xs2->setTotalXS(4, 0.3, 5.0 );
        xs2->setTotalXS(5, 0.4, 6.0 );
        xs2->setTotalXS(6, 0.5, 7.0 );
        xs2->setTotalXS(7, 0.7, 8.0 );
        xs2->setTotalXS(8, 0.9, 9.0 );
        xs2->setTotalXS(9, 1.0, 10.0 );

        HashLookupHost* hash = new HashLookupHost(4, 10);
        hash->addIsotope( xs );
        hash->addIsotope( xs2 );

        gpuFloatType_t energy = 2.25;
        CHECK_CLOSE( 3.5, getTotalXS(xs->getXSPtr(),energy), 1e-5);
        unsigned HashBin = getHashBin(hash->getPtr(),energy);
        gpuFloatType_t value = getTotalXS(xs->getXSPtr(), hash->getPtr(), HashBin, energy);
        CHECK_CLOSE( 3.5, value, 1e-6);

        delete xs;
        delete xs2;
        delete hash;
    }

    TEST( hashlookup_load_u235_from_file)
    {
        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(1);
        xs->read( "MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin");

        gpuFloatType_t energy = 2.0;

        CHECK_EQUAL( 76525, xs->size() );
        CHECK_CLOSE( 233.025, xs->getAWR(), 1e-3 );
        double value = getTotalXS(xs->getXSPtr(), energy);
        CHECK_CLOSE( 7.14769f, value, 1e-5);

        HashLookupHost* hash = new HashLookupHost(1);
        hash->addIsotope( xs );
        unsigned HashBin = getHashBin(hash->getPtr(),energy);
        value = getTotalXS(xs->getXSPtr(), hash->getPtr(), HashBin, energy);
        CHECK_CLOSE( 7.14769f, value, 1e-5);

        delete xs;
        delete hash;
    }

    TEST( write_read ){
         MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(1);
         xs->read( "MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin");

         gpuFloatType_t energy = 2.0;

         CHECK_EQUAL( 76525, xs->size() );
         CHECK_CLOSE( 233.025, xs->getAWR(), 1e-3 );
         double value = getTotalXS(xs->getXSPtr(), energy);
         CHECK_CLOSE( 7.14769f, value, 1e-5);

         HashLookupHost write_hash(1);
         write_hash.addIsotope( xs );
         write_hash.writeToFile( "crosssection_hash_write_test1.bin" );

         // test file exists
         std::ifstream exists("crosssection_hash_write_test1.bin");
         CHECK_EQUAL( true, exists.good() );
         exists.close();

         HashLookupHost hash(1);
         hash.readFromFile( "crosssection_hash_write_test1.bin" );

         unsigned HashBin = getHashBin(hash.getPtr(),energy);
         value = getTotalXS(xs->getXSPtr(), hash.getPtr(), HashBin, energy);
         CHECK_CLOSE( 7.14769f, value, 1e-5);

         delete xs;
     }

#if false
    // Currently the hash table is not used for photon cross-sections
    TEST( hashlookup_load_photon_Uranium_from_file)
    {
        MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(1);
        xs->read( "MonteRayTestFiles/92000-04p_MonteRayCrossSection.bin");

        gpuFloatType_t energy = 1.0;

        CHECK_EQUAL( 504, xs->size() );
        CHECK_CLOSE( 235.984, xs->getAWR(), 1e-3 );
        double value = getTotalXS(xs->getXSPtr(), energy);
        CHECK_CLOSE( 30.9887f, value, 1e-4);

        HashLookupHost* hash = new HashLookupHost(1);
        hash->addIsotope( xs );
        unsigned HashBin = getHashBin(hash->getPtr(),energy);
        value = getTotalXS(xs->getXSPtr(), hash->getPtr(), HashBin, energy);
        CHECK_CLOSE( 7.14769f, value, 1e-5);

        delete xs;
        delete hash;
    }
#endif

    //    TEST_FIXTURE(MonteRayCrossSectionTestHelper, get_total_xs_from_gpu ) {
    //    	MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(4);
    //    	xs->setTotalXS(0, 0.0, 4.0 );
    //    	xs->setTotalXS(1, 1.0, 3.0 );
    //    	xs->setTotalXS(2, 2.0, 2.0 );
    //    	xs->setTotalXS(3, 3.0, 1.0 );
    //
    //    	xs->copyToGPU();
    //
    //    	gpuFloatType_t energy = 0.5;
    //
    //    	setupTimers();
    //    	gpuFloatType_t totalXS = launchGetTotalXS( xs, energy);
    //    	stopTimers();
    //
    //    	CHECK_CLOSE( 3.5f, totalXS, 1e-7 );
    //
    //    	delete xs;
    //    }
    //
    //    TEST_FIXTURE(MonteRayCrossSectionTestHelper, load_u235_from_file)
    //    {
    //    	MonteRayCrossSectionHost* xs = new MonteRayCrossSectionHost(1);
    //    	xs->read( "MonteRayTestFiles/u235_simpleCrossSection.bin");
    //
    //    	gpuFloatType_t energy = 2.0;
    //
    //    	CHECK_EQUAL( 24135, xs->size() );
    //    	CHECK_CLOSE( 233.025, xs->getAWR(), 1e-3 );
    //    	double value = getTotalXS(xs->getXSPtr(), energy);
    //    	CHECK_CLOSE( 7.17639378000f, value, 1e-6);
    //
    //    	xs->copyToGPU();
    //
    //    	gpuSync sync;
    //    	gpuFloatType_t totalXS = launchGetTotalXS( xs, energy);
    //    	sync.sync();
    //
    //    	CHECK_CLOSE( 7.17639378000f, totalXS, 1e-7 );
    //
    //    	delete xs;
    //    }

}
