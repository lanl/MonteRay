#include <UnitTest++.h>

#include <vector>
#include <sstream>
#include <random>
#include <memory>

#ifdef __CUDACC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif

#include "CrossSection.hh"

#include "GPUUtilityFunctions.hh"
#include "MonteRay_timer.hh"
#include "ReadAndWriteFiles.hh"

namespace CrossSection_tester_namespace {

using namespace MonteRay;

SUITE( CrossSection_tester ) {

    TEST( build_with_energy_and_xsec_vectors ) {
        std::vector<double> energies = {0, 1, 2, 3};
        std::vector<double> xsecs = {4, 3, 2, 1};
        int ZAID = 1001;

        CrossSection XS = CrossSectionBuilder( ZAID, energies, xsecs ).build();
        CHECK_EQUAL( 4, XS.size() );
        CHECK_EQUAL( 1001, XS.ZAID() );

        CHECK_CLOSE( 0.0, XS.getEnergy(0), 1e-5 );
        CHECK_CLOSE( 1.0, XS.getEnergy(1), 1e-5 );
        CHECK_CLOSE( 2.0, XS.getEnergy(2), 1e-5 );
        CHECK_CLOSE( 3.0, XS.getEnergy(3), 1e-5 );

        CHECK_CLOSE( 4.0, XS.getTotalXSByIndex(0), 1e-5 );
        CHECK_CLOSE( 3.0, XS.getTotalXSByIndex(1), 1e-5 );
        CHECK_CLOSE( 2.0, XS.getTotalXSByIndex(2), 1e-5 );
        CHECK_CLOSE( 1.0, XS.getTotalXSByIndex(3), 1e-5 );
    }

    CUDA_CALLABLE_KERNEL kernelGetSize( CrossSection* xs, int* value) {
        int size = xs->size();
        value[0] = size;
    }

    TEST( CrossSection_size_on_GPU ) {
        managed_vector<int> value;
        value.push_back( -1 );

        std::vector<double> energies = {0, 1, 2, 3};
        std::vector<double> xsecs = {4, 3, 2, 1};
        int ZAID = 1001;

        auto pXS = std::make_unique<CrossSection>( CrossSectionBuilder( ZAID, energies, xsecs ).build() );

        CHECK_EQUAL( 4, pXS->size() );

        CHECK_EQUAL( -1, value[0] );

#ifdef __CUDACC__
        kernelGetSize<<<1,1>>>( pXS.get(), value.data() );
        cudaDeviceSynchronize();
#else
        kernelGetSize( pXS.get(), value.data() );
#endif
        CHECK_EQUAL( 4, pXS->size() );

        CHECK_EQUAL( 4, value[0] );
    }

    TEST( CrossSection_set_particle_type ) {
        std::vector<double> energies1 = {0, 1, 2, 3};
        std::vector<double> xsecs1 = {4, 3, 2, 1};
        int ZAID = 1001;

        CrossSectionBuilder xsbuilder( ZAID, energies1, xsecs1 );
        xsbuilder.setParticleType( photon );

        CrossSection xs = xsbuilder.build();
        CHECK_EQUAL( photon, xs.getParticleType() );
    }

    TEST( CrossSection_read_write ) {
        std::vector<double> energies1 = {0, 1, 2, 3};
        std::vector<double> xsecs1 = {4, 3, 2, 1};
        int ZAID = 1001;

        CrossSectionBuilder xsbuilder( ZAID, energies1, xsecs1, photon, 1.33 );
        xsbuilder.setParticleType( photon );

        CrossSection xs = xsbuilder.build();

        std::stringstream ss;
        xs.write(ss);

        auto xs2 = CrossSection::read(ss);

        CHECK_EQUAL( 1001, xs2.ZAID() );
        CHECK_EQUAL( 4, xs2.size() );
        CHECK_EQUAL( photon, xs2.getParticleType() );
        CHECK_CLOSE( 1.33, xs2.getAWR(), 1e-5 );

        CHECK_CLOSE( 0.0, xs2.getEnergy(0), 1e-5 );
        CHECK_CLOSE( 1.0, xs2.getEnergy(1), 1e-5 );
        CHECK_CLOSE( 2.0, xs2.getEnergy(2), 1e-5 );
        CHECK_CLOSE( 3.0, xs2.getEnergy(3), 1e-5 );

        CHECK_CLOSE( 4.0, xs2.getTotalXSByIndex(0), 1e-5 );
        CHECK_CLOSE( 3.0, xs2.getTotalXSByIndex(1), 1e-5 );
        CHECK_CLOSE( 2.0, xs2.getTotalXSByIndex(2), 1e-5 );
        CHECK_CLOSE( 1.0, xs2.getTotalXSByIndex(3), 1e-5 );
    }

    TEST( CrossSection_getIndex_as_function_of_energy ) {
        std::vector<double> energies1 = {0.2, 1, 2, 3};
        std::vector<double> xsecs1 = {4, 3, 2, 1};
        int ZAID = 1001;

        auto xs = CrossSection( ZAID, energies1, xsecs1, photon, 1.33 );

        CHECK_EQUAL( 0, xs.getIndex(0.1) );
        CHECK_EQUAL( 0, xs.getIndex(0.5) );
        CHECK_EQUAL( 1, xs.getIndex(1.5) );
        CHECK_EQUAL( 2, xs.getIndex(2.5) );
        CHECK_EQUAL( 3, xs.getIndex(3.5) );
    }

    TEST( CrossSection_getTotalXS_as_function_of_energy ) {
        std::vector<double> energies1 = {0.2, 1, 2, 3};
        std::vector<double> xsecs1 = {4, 3, 2, 1};
        int ZAID = 1001;

        auto xs = CrossSection( ZAID, energies1, xsecs1, photon, 1.33 );

        CHECK_CLOSE( 4.0, xs.getTotalXS(0.1), 1e-5 );
        CHECK_CLOSE( 3.5, xs.getTotalXS(0.6), 1e-5 );
        CHECK_CLOSE( 2.5, xs.getTotalXS(1.5), 1e-5 );
        CHECK_CLOSE( 1.5, xs.getTotalXS(2.5), 1e-5 );
        CHECK_CLOSE( 1.0, xs.getTotalXS(3.5), 1e-5 );
    }

    TEST( read_neutron_file ) {

        auto xs = readFromFile<CrossSection>( std::string("MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin") );

        CHECK_EQUAL( 76525, xs.size() );
        CHECK_CLOSE( 233.025, xs.getAWR(), 1e-3 );

        gpuFloatType_t energy = 2.0;
        double value = xs.getTotalXS(energy);
        CHECK_CLOSE( 7.14769f, value, 1e-5);
    }

    CUDA_CALLABLE_KERNEL kernelTotalXS( CrossSection* xs, gpuFloatType_t* value, gpuFloatType_t E) {
        gpuFloatType_t totalXS = xs->getTotalXS(E);
        value[0] = totalXS;
    }

    CUDA_CALLABLE_KERNEL kernelTotalXSByIndex( CrossSection* xs, gpuFloatType_t* value, size_t i) {
        gpuFloatType_t totalXS = xs->getTotalXSByIndex(i);
        value[0] = totalXS;
    }

    CUDA_CALLABLE_KERNEL kernelGetIndex( CrossSection* xs, size_t* value, gpuFloatType_t E) {
        size_t index = xs->getIndex(E);
        value[0] = index;
    }

    TEST( Test_U235_on_GPU ) {
        managed_vector<size_t> index_value;
        index_value.push_back( 0 );

        managed_vector<int> size_value;
        size_value.push_back( 0 );

        managed_vector<gpuFloatType_t> xs_value;
        xs_value.push_back( 0.0 );

        managed_vector<gpuFloatType_t> xsbyIndex_value;
        xsbyIndex_value.push_back( 0.0 );

        auto pXS = std::make_unique<CrossSection>(
          readFromFile<CrossSection>( std::string("MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin") )
        );

        deviceSynchronize();

#ifdef __CUDACC__
        kernelGetSize<<<1,1>>>( pXS.get(), size_value.data() );
        cudaDeviceSynchronize();
#else
        kernelGetSize( pXS.get(), size_value.data() );
#endif
        CHECK_EQUAL( 76525, size_value[0] );

        gpuFloatType_t energy = 2.0;
#ifdef __CUDACC__
        kernelTotalXS<<<1,1>>>( pXS.get(), xs_value.data(), energy );
        cudaDeviceSynchronize();
#else
        kernelTotalXS( pXS.get(), xs_value.data(), energy );
#endif

        CHECK_CLOSE( 7.14769f, pXS->getTotalXS(energy), 1e-5);
        CHECK_CLOSE( 7.14769f, xs_value[0], 1e-5);

#ifdef __CUDACC__
        kernelGetIndex<<<1,1>>>( pXS.get(), index_value.data(), energy );
        cudaDeviceSynchronize();
#else
        kernelGetIndex( pXS.get(), index_value.data(), energy );
#endif

        CHECK_EQUAL( 76420, index_value[0] );
        CHECK_EQUAL( 76420, pXS->getIndex(energy) );

#ifdef __CUDACC__
        kernelTotalXSByIndex<<<1,1>>>( pXS.get(), xsbyIndex_value.data(), index_value[0] );
        cudaDeviceSynchronize();
#else
        kernelTotalXSByIndex( pXS.get(), xsbyIndex_value.data(),  pXS->getIndex(energy) );
#endif

        CHECK_CLOSE( 7.14769f, pXS->getTotalXSByIndex(pXS->getIndex(energy)), 1e-5);
        CHECK_CLOSE( 7.14769f, xsbyIndex_value[0], 1e-5);

#ifdef __CUDACC__
        kernelTotalXSByIndex<<<1,1>>>( pXS.get(), xsbyIndex_value.data(), index_value[0]+1 );
        cudaDeviceSynchronize();
#else
        kernelTotalXSByIndex( pXS.get(), xsbyIndex_value.data(),  pXS->getIndex(energy)+1 );
#endif

        CHECK_CLOSE( 7.28222f, pXS->getTotalXSByIndex(pXS->getIndex(energy)+1), 1e-5);
        CHECK_CLOSE( 7.28222f, xsbyIndex_value[0], 1e-5);

    }

    class grid {
    public:
        grid( std::vector<double> values ) {
            grid_values = values;
        }

        unsigned GridSize(void) const { return grid_values.size();}
        double operator[](unsigned i) const { return grid_values[i];}
        std::vector<double> grid_values;
    };

    class neutronXSTester {
    public:
        std::string getType() const { return "neutron"; }

        unsigned index( double E ) const {
            auto lower = std::lower_bound( energyBins.begin(), energyBins.end(), E );
            unsigned index = std::distance( energyBins.begin(), lower );
            if( index > 0 ) --index;
            return index;
        }

        double TotalXsec( size_t i ) const {
            return xsValues[i];
        }

        double TotalXsec( unsigned i ) const {
            return xsValues[i];
        }

        double TotalXsec(double E, double T, unsigned i ) const {
            MONTERAY_ASSERT(i < energyBins.size());
            double frac = ( E - energyBins[i] ) / (energyBins[i+1] - energyBins[i]);
            return xsValues[i] + (xsValues[i+1] - xsValues[i] ) * frac;
        }

        double TotalXsec(double E, double T = -1.0 ) const {
          double lowest = energyBins.front();
          double highest = energyBins.back();
          if( E <=  lowest ) {
              return xsValues.front();
          }
          if( E >=  highest ) {
              return xsValues.back();
          }
          unsigned i = index(E);
          return TotalXsec( E, T, i);
        }

        grid getEnergyGrid() const { return grid(energyBins); }
        double getAWR() const { return 1.1;}
        unsigned getZAID() const { return 1001; }

        std::vector<double> energyBins = { 0.0, 1.0, 2.0, 3.0 };
        std::vector<double> xsValues   = { 4.0, 3.0, 2.0, 1.0 };
    };

    TEST( neutronXSTester_test ) {
        neutronXSTester xs;

        CHECK_EQUAL( 0, xs.index( -1.0 ) );
        CHECK_EQUAL( 0, xs.index( 0.5 ) );
        CHECK_EQUAL( 0, xs.index( 1.0 ) );
        CHECK_EQUAL( 1, xs.index( 1.5 ) );
        CHECK_EQUAL( 3, xs.index( 3.5 ) );

        CHECK_CLOSE( 4.0, xs.TotalXsec( 0.0 ), 1e-6 );
        CHECK_CLOSE( 3.5, xs.TotalXsec( 0.5 ), 1e-6 );
        CHECK_CLOSE( 3.0, xs.TotalXsec( 1.0 ), 1e-6 );
        CHECK_CLOSE( 2.5, xs.TotalXsec( 1.5 ), 1e-6 );
        CHECK_CLOSE( 1.5, xs.TotalXsec( 2.5 ), 1e-6 );
        CHECK_CLOSE( 1.0, xs.TotalXsec( 3.0 ), 1e-6 );
        CHECK_CLOSE( 1.0, xs.TotalXsec( 3.5 ), 1e-6 );
    }

    TEST( build_XS_template ) {
        neutronXSTester xs;

        CrossSection XS = CrossSectionBuilder( xs ).build();
        CHECK_EQUAL( 4, XS.size() );
        CHECK_EQUAL( 1001, XS.ZAID() );
        CHECK_CLOSE( 1.1, XS.getAWR(), 1e-5 );
    }

}

//TEST( read_neutron_file ) {
//
//    CrossSectionBuilder xsbuilder;
//    xsbuilder.readFromFile( std::string("MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin") )
//
//    CrossSection xs = xsbuilder.build();
//
//    CHECK_EQUAL( 76525, xs.size() );
//    CHECK_CLOSE( 233.025, xs.getAWR(), 1e-3 );
//
//    gpuFloatType_t energy = 2.0;
//    double value = xs.getTotalXS(energy);
//    CHECK_CLOSE( 7.14769f, value, 1e-5);
//}

SUITE( CrossSection_speed_tester ) {

    template<typename EXECUTION_POLICY, typename HASHFUNCTION = FasterHash>
    void transformEnergyToXS( EXECUTION_POLICY policy,
                              managed_vector<gpuFloatType_t>& energies,
                              managed_vector<gpuFloatType_t>& results,
                              CrossSection_t<HASHFUNCTION>* xs ) {
        auto calcXS = [=]  CUDA_CALLABLE_MEMBER (gpuFloatType_t E) { return xs->getTotalXS( E ); };

#ifdef __CUDACC__
        thrust::transform( policy, energies.data(), energies.data()+energies.size(),
                results.data(), calcXS );
#else
        std::transform( energies.data(), energies.data()+energies.size(),
                        results.data(), calcXS );
#endif

    }

    template<typename EXECUTION_POLICY, typename HASHFUNCTION = FasterHash>
    void transformEnergyToXSViaHash( EXECUTION_POLICY policy,
            managed_vector<gpuFloatType_t>& energies,
            managed_vector<gpuFloatType_t>& results,
            CrossSection_t<HASHFUNCTION>* xs ) {
        auto calcXS = [=]  CUDA_CALLABLE_MEMBER (gpuFloatType_t E) { return xs->getTotalXSviaHash( E ); };

#ifdef __CUDACC__
        thrust::transform( policy, energies.data(), energies.data()+energies.size(),
                results.data(), calcXS );
#else
        std::transform( energies.data(), energies.data()+energies.size(),
                        results.data(), calcXS );
#endif

    }

    TEST( neutron_lookup_speed_cpu ) {
        // 1e7 lookups =  0.4806 seconds (release) - Binary Lookup CPU

        managed_vector<gpuFloatType_t> energies;
        managed_vector<gpuFloatType_t> xsecs;
        unsigned nEnergies = 10000000;
        energies.resize( nEnergies );
        xsecs.resize( nEnergies );

        typedef std::mersenne_twister_engine<std::uint_fast32_t, 32, 624, 397, 31,
                                     0x9908b0df, 11,
                                     0xffffffff, 7,
                                     0x9d2c5680, 15,
                                     0xefc60000, 18, 1812433253> mt19937;

        mt19937::result_type seed = 1001;
        auto real_rand = std::bind(std::uniform_real_distribution<gpuFloatType_t>(0,1),
                mt19937(seed));


        for( unsigned i = 0; i < nEnergies; ++i ) {
            energies[i] = real_rand();
        }

        auto pXS = std::make_unique<CrossSection>(
          readFromFile<CrossSection>( std::string("MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin") )
        );

        std::cout << "Debug: Starting U-235 neutron cross-section timing lookup test\n";
        cpuTimer timer;
        timer.start();

#ifdef __CUDACC__
        auto policy = thrust::host;
#else
        auto policy = false;
#endif

        transformEnergyToXS( policy, energies, xsecs, pXS.get() );
        timer.stop();
        std::cout << "Debug: Time to lookup " << nEnergies << " U-235 neutron cross-sections = " << timer.getTime() <<
                " seconds\n";
    }

    CUDA_CALLABLE_KERNEL kernelAllTotalXS( CrossSection* xs, gpuFloatType_t* values, gpuFloatType_t* energies, unsigned num) {

#ifdef __CUDACC__
        unsigned threadID = threadIdx.x + blockIdx.x*blockDim.x;

        unsigned particleID = threadID;

        while( particleID < num ) {
            values[particleID] = xs->getTotalXS( energies[particleID] );
            particleID += blockDim.x*gridDim.x;
        }
#else
        for( unsigned particleID = 0; particleID<num; ++particleID) {
            values[particleID] = xs->getTotalXS( energies[particleID] );
        }
#endif

    }

    TEST( neutron_lookup_speed_gpu ) {
         // 1e7 lookups =  0.4806 seconds (release) - Binary Lookup CPU
         // 1e7 lookups =  0.02139 seconds (release) - Binary Lookup GPU

         managed_vector<gpuFloatType_t> ref_xsecs;
         managed_vector<gpuFloatType_t> xsecs;
         managed_vector<gpuFloatType_t> energies;
         unsigned nEnergies = 10000000;
         energies.resize( nEnergies );
         xsecs.resize( nEnergies );
         ref_xsecs.resize( nEnergies );

         typedef std::mersenne_twister_engine<std::uint_fast32_t, 32, 624, 397, 31,
                                      0x9908b0df, 11,
                                      0xffffffff, 7,
                                      0x9d2c5680, 15,
                                      0xefc60000, 18, 1812433253> mt19937;

         mt19937::result_type seed = 1001;
         auto real_rand = std::bind(std::uniform_real_distribution<gpuFloatType_t>(0,1),
                 mt19937(seed));


         for( unsigned i = 0; i < nEnergies; ++i ) {
             energies[i] = real_rand();
         }

        auto pXS = std::make_unique<CrossSection>(
          readFromFile<CrossSection>( std::string("MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin") )
        );

         deviceSynchronize();

        // auto calcXS = [=]  __host__ __device__ (double E) { return xs->getTotalXS( E ); };

         std::cout << "Debug: Starting U-235 neutron cross-section timing lookup test on GPU with sort\n";
         cpuTimer timer;
         timer.start();

#ifdef __CUDACC__
         thrust::sort( thrust::device, energies.data(), energies.data()+energies.size() );
#else
         std::sort( energies.data(), energies.data()+energies.size() );
#endif

#ifdef __CUDACC__
        auto device_policy = thrust::device;
#else
        auto device_policy = false;
#endif

         //kernelAllTotalXS<<<1024,512>>>( xs, xsecs.data(), energies.data(), energies.size() );
         transformEnergyToXS( device_policy, energies, xsecs, pXS.get() );

         deviceSynchronize();
         timer.stop();
         std::cout << "Debug: Time to lookup " << nEnergies << " U-235 neutron cross-sections = " << timer.getTime() <<
                 " seconds\n";

#ifdef __CUDACC__
        auto host_policy = thrust::host;
#else
        auto host_policy = false;
#endif

         transformEnergyToXS( host_policy, energies, ref_xsecs, pXS.get() );

         for( unsigned i = 0; i < energies.size(); ++i ){
             gpuFloatType_t percent_diff = (ref_xsecs[i] - xsecs[i] ) / ref_xsecs[i];
             CHECK_CLOSE(0.0, percent_diff, 1e-5 );

             // only report one error
             if( std::abs( percent_diff) > 1e-5  ) {
                 break;
             }
         }
     }

    TEST( hash_lookup_vs_standard_lookup ) {
         // 1e7 lookups =  0.4806 seconds (release) - Binary Lookup CPU
         // 1e7 lookups =  0.02139 seconds (release) - Binary Lookup GPU

         managed_vector<gpuFloatType_t> ref_xsecs;
         managed_vector<gpuFloatType_t> xsecs;
         managed_vector<gpuFloatType_t> energies;
         unsigned nEnergies = 10000000;
         energies.resize( nEnergies );
         xsecs.resize( nEnergies );
         ref_xsecs.resize( nEnergies );

         typedef std::mersenne_twister_engine<std::uint_fast32_t, 32, 624, 397, 31,
                                      0x9908b0df, 11,
                                      0xffffffff, 7,
                                      0x9d2c5680, 15,
                                      0xefc60000, 18, 1812433253> mt19937;

         mt19937::result_type seed = 1001;
         auto real_rand = std::bind(std::uniform_real_distribution<gpuFloatType_t>(0,1),
                 mt19937(seed));


         for( unsigned i = 0; i < nEnergies; ++i ) {
             energies[i] = real_rand();
         }

         auto pXS = std::make_unique<CrossSection>(
           readFromFile<CrossSection>( std::string("MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin") )
         );
//         cudaDeviceSynchronize();
//
//        // auto calcXS = [=]  __host__ __device__ (double E) { return xs->getTotalXS( E ); };
//
//         std::cout << "Debug: Starting U-235 neutron cross-section timing lookup test on GPU\n";
//         cpuTimer timer;
//         timer.start();
//         thrust::sort( thrust::device, energies.data(), energies.data()+energies.size() );
//         //kernelAllTotalXS<<<1024,512>>>( xs, xsecs.data(), energies.data(), energies.size() );
//         transformEnergyToXS( thrust::device, energies, xsecs, xs );
//
//         cudaDeviceSynchronize();
//         timer.stop();
//         std::cout << "Debug: Time to lookup " << nEnergies << " U-235 neutron cross-sections = " << timer.getTime() <<
//                 " seconds\n";

#ifdef __CUDACC__
        auto policy = thrust::host;
#else
        auto policy = false;
#endif

         transformEnergyToXS( policy, energies, ref_xsecs, pXS.get() );

         std::cout << "Debug: Starting U-235 neutron cross-section timing lookup test on CPU via hash lookup\n";
         cpuTimer timer;
         timer.start();
         transformEnergyToXSViaHash( policy, energies, xsecs, pXS.get() );
         timer.stop();
         std::cout << "Debug: Time to perform hash lookup of " << nEnergies << " U-235 neutron cross-sections = " << timer.getTime() <<
                      " seconds\n";
         std::cout << "Debug: Complete\n";

         for( unsigned i = 0; i < energies.size(); ++i ){
             gpuFloatType_t percent_diff = (ref_xsecs[i] - xsecs[i] ) / ref_xsecs[i];
             CHECK_CLOSE(0.0, percent_diff, 1e-5 );

             // only report one error
             if( std::abs( percent_diff) > 1e-5  ) {
                 break;
             }
         }
     }

    class testLog2Hash {

    public:

        CUDA_CALLABLE_MEMBER
        static size_t hashFunction(const double value) {
            // For double
            // shifts the bits and returns binary equivalent integer
            //std::cout << "Debug -- Calling hashFunction(double)\n";

            MONTERAY_ASSERT_MSG( value >= 0.0, "Negative values are not allowed.");

            std::uint64_t i = (( std::log2(value)+40.0)*100.0);
            return i;
        }

        CUDA_CALLABLE_MEMBER
        static size_t hashFunction(const float value) {
            // For float
            // shifts the bits and returns binary equivalent integer
            //std::cout << "Debug -- Calling hashFunction(float)\n";

            MONTERAY_ASSERT_MSG( value >= 0.0f, "Negative values are not allowed.");

            std::uint64_t i = (( std::log2(value)+40.0f)*100.0f);
            return i;
        }

        template < typename TV, typename std::enable_if< sizeof(TV) == 8 >::type* = nullptr >
        CUDA_CALLABLE_MEMBER
        static
        TV invHashFunction( const size_t index, const size_t minIndex = 0 ) {
            //std::cout << "Debug -- Calling invHashFunction(index)->double\n";

            TV value = std::exp2( ((index + minIndex) / 100.0 ) - 40.0 );
            return value;
        }

        template < typename TV, typename std::enable_if< sizeof(TV) == 4 >::type* = nullptr >
        CUDA_CALLABLE_MEMBER
        static
        TV invHashFunction( const size_t index, const size_t minIndex = 0 ) {
            //std::cout << "Debug -- Calling invHashFunction(index)->float\n";

            TV value = std::exp2( ((index + minIndex) / 100.0f ) - 40.0f );
            return value;
        }

    };

    TEST( hashlog2_lookup_vs_standard_lookup ) {
          // 1e7 lookups =  0.4806 seconds (release) - Binary Lookup CPU
          // 1e7 lookups =  0.02139 seconds (release) - Binary Lookup GPU

          managed_vector<gpuFloatType_t> ref_xsecs;
          managed_vector<gpuFloatType_t> xsecs;
          managed_vector<gpuFloatType_t> energies;
          unsigned nEnergies = 10000000;
          energies.resize( nEnergies );
          xsecs.resize( nEnergies );
          ref_xsecs.resize( nEnergies );

          typedef std::mersenne_twister_engine<std::uint_fast32_t, 32, 624, 397, 31,
                                       0x9908b0df, 11,
                                       0xffffffff, 7,
                                       0x9d2c5680, 15,
                                       0xefc60000, 18, 1812433253> mt19937;

          mt19937::result_type seed = 1001;
          auto real_rand = std::bind(std::uniform_real_distribution<gpuFloatType_t>(0,1),
                  mt19937(seed));


          for( unsigned i = 0; i < nEnergies; ++i ) {
              energies[i] = real_rand();
          }

          using CrossSection = CrossSection_t<testLog2Hash>;
          auto pXS = std::make_unique<CrossSection>(
            readFromFile<CrossSection>( std::string("MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin") )
          );
 //         cudaDeviceSynchronize();
 //
 //        // auto calcXS = [=]  __host__ __device__ (double E) { return xs->getTotalXS( E ); };
 //
 //         std::cout << "Debug: Starting U-235 neutron cross-section timing lookup test on GPU\n";
 //         cpuTimer timer;
 //         timer.start();
 //         thrust::sort( thrust::device, energies.data(), energies.data()+energies.size() );
 //         //kernelAllTotalXS<<<1024,512>>>( xs, xsecs.data(), energies.data(), energies.size() );
 //         transformEnergyToXS( thrust::device, energies, xsecs, xs );
 //
 //         cudaDeviceSynchronize();
 //         timer.stop();
 //         std::cout << "Debug: Time to lookup " << nEnergies << " U-235 neutron cross-sections = " << timer.getTime() <<
 //                 " seconds\n";

 #ifdef __CUDACC__
         auto policy = thrust::host;
 #else
         auto policy = false;
 #endif

          transformEnergyToXS( policy, energies, ref_xsecs, pXS.get() );

          std::cout << "Debug: Starting U-235 neutron cross-section timing lookup test on CPU via log2 hash lookup\n";
          cpuTimer timer;
          timer.start();
          transformEnergyToXSViaHash( policy, energies, xsecs, pXS.get() );
          timer.stop();
          std::cout << "Debug: Time to perform hash log2 lookup of " << nEnergies << " U-235 neutron cross-sections = " << timer.getTime() <<
                       " seconds\n";
          std::cout << "Debug: Complete\n";

          for( unsigned i = 0; i < energies.size(); ++i ){
              gpuFloatType_t percent_diff = (ref_xsecs[i] - xsecs[i] ) / ref_xsecs[i];
              CHECK_CLOSE(0.0, percent_diff, 1e-5 );

              // only report one error
              if( std::abs( percent_diff) > 1e-5  ) {
                  break;
              }
          }
      }

    class testLogHash {

    public:

        CUDA_CALLABLE_MEMBER
        static size_t hashFunction(const double value) {
            // For double
            // shifts the bits and returns binary equivalent integer
            //std::cout << "Debug -- Calling hashFunction(double)\n";

            MONTERAY_ASSERT_MSG( value >= 0.0, "Negative values are not allowed.");

            std::uint64_t i = (( std::log(value)+30.0)*200.0);
            return i;
        }

        CUDA_CALLABLE_MEMBER
        static size_t hashFunction(const float value) {
            // For float
            // shifts the bits and returns binary equivalent integer
            //std::cout << "Debug -- Calling hashFunction(float)\n";

            MONTERAY_ASSERT_MSG( value >= 0.0f, "Negative values are not allowed.");

            std::uint64_t i = (( std::log(value)+30.0f)*200.0f);
            return i;
        }

        template < typename TV, typename std::enable_if< sizeof(TV) == 8 >::type* = nullptr >
        CUDA_CALLABLE_MEMBER
        static
        TV invHashFunction( const size_t index, const size_t minIndex = 0 ) {
            //std::cout << "Debug -- Calling invHashFunction(index)->double\n";

            TV value = std::exp( ((index + minIndex) / 200.0 ) - 30.0 );
            return value;
        }

        template < typename TV, typename std::enable_if< sizeof(TV) == 4 >::type* = nullptr >
        CUDA_CALLABLE_MEMBER
        static
        TV invHashFunction( const size_t index, const size_t minIndex = 0 ) {
            //std::cout << "Debug -- Calling invHashFunction(index)->float\n";

            TV value = std::exp( ((index + minIndex) / 200.0f ) - 30.0f );
            return value;
        }

    };

    TEST( hashlog_lookup_vs_standard_lookup ) {
          // 1e7 lookups =  0.4806 seconds (release) - Binary Lookup CPU
          // 1e7 lookups =  0.02139 seconds (release) - Binary Lookup GPU

          managed_vector<gpuFloatType_t> ref_xsecs;
          managed_vector<gpuFloatType_t> xsecs;
          managed_vector<gpuFloatType_t> energies;
          unsigned nEnergies = 10000000;
          energies.resize( nEnergies );
          xsecs.resize( nEnergies );
          ref_xsecs.resize( nEnergies );

          typedef std::mersenne_twister_engine<std::uint_fast32_t, 32, 624, 397, 31,
                                       0x9908b0df, 11,
                                       0xffffffff, 7,
                                       0x9d2c5680, 15,
                                       0xefc60000, 18, 1812433253> mt19937;

          mt19937::result_type seed = 1001;
          auto real_rand = std::bind(std::uniform_real_distribution<gpuFloatType_t>(0,1),
                  mt19937(seed));


          for( unsigned i = 0; i < nEnergies; ++i ) {
              energies[i] = real_rand();
          }

          using CrossSection = CrossSection_t<testLogHash>;
          auto pXS = std::make_unique<CrossSection>(
            readFromFile<CrossSection>( std::string("MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin") )
          );
 //         cudaDeviceSynchronize();
 //
 //        // auto calcXS = [=]  __host__ __device__ (double E) { return xs->getTotalXS( E ); };
 //
 //         std::cout << "Debug: Starting U-235 neutron cross-section timing lookup test on GPU\n";
 //         cpuTimer timer;
 //         timer.start();
 //         thrust::sort( thrust::device, energies.data(), energies.data()+energies.size() );
 //         //kernelAllTotalXS<<<1024,512>>>( xs, xsecs.data(), energies.data(), energies.size() );
 //         transformEnergyToXS( thrust::device, energies, xsecs, xs );
 //
 //         cudaDeviceSynchronize();
 //         timer.stop();
 //         std::cout << "Debug: Time to lookup " << nEnergies << " U-235 neutron cross-sections = " << timer.getTime() <<
 //                 " seconds\n";

 #ifdef __CUDACC__
         auto policy = thrust::host;
 #else
         auto policy = false;
 #endif

          transformEnergyToXS( policy, energies, ref_xsecs, pXS.get() );

          std::cout << "Debug: Starting U-235 neutron cross-section timing lookup test on CPU via log hash lookup\n";
          cpuTimer timer;
          timer.start();
          transformEnergyToXSViaHash( policy, energies, xsecs, pXS.get() );
          timer.stop();
          std::cout << "Debug: Time to perform hash log lookup of " << nEnergies << " U-235 neutron cross-sections = " << timer.getTime() <<
                       " seconds\n";
          std::cout << "Debug: Complete\n";

          for( unsigned i = 0; i < energies.size(); ++i ){
              gpuFloatType_t percent_diff = (ref_xsecs[i] - xsecs[i] ) / ref_xsecs[i];
              CHECK_CLOSE(0.0, percent_diff, 1e-5 );

              // only report one error
              if( std::abs( percent_diff) > 1e-5  ) {
                  break;
              }
          }
      }

}

} // end namespace
