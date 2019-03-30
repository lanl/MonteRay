#include <UnitTest++.h>

#include <vector>
#include <sstream>
#include <random>

#ifdef __CUDACC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif

#include "CrossSection.hh"

#include "GPUUtilityFunctions.hh"
#include "MonteRay_timer.hh"

namespace CrossSection_tester_namespace {

using namespace MonteRay;

SUITE( CrossSection_tester ) {

    TEST( build_with_energy_and_xsec_vectors ) {
        std::vector<double> energies = {0, 1, 2, 3};
        std::vector<double> xsecs = {4, 3, 2, 1};
        int ZAID = 1001;

        CrossSection XS = CrossSectionBuilder( ZAID, energies, xsecs ).construct();
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

        CrossSection* XS = new CrossSection;
        //cudaMallocManaged(&XS, sizeof(CrossSection) );

        //managed_allocator<CrossSection> alloc;
        //auto XS = std::allocate_shared<CrossSection>(alloc);
        *XS = CrossSectionBuilder( ZAID, energies, xsecs ).construct();

        CHECK_EQUAL( 4, XS->size() );

        CHECK_EQUAL( -1, value[0] );

#ifdef __CUDACC__
        kernelGetSize<<<1,1>>>( XS, value.data() );
        cudaDeviceSynchronize();
#else
        kernelGetSize( XS, value.data() );
#endif
        CHECK_EQUAL( 4, XS->size() );

        CHECK_EQUAL( 4, value[0] );
        delete XS;
    }

    TEST( CrossSection_set_particle_type ) {
        std::vector<double> energies1 = {0, 1, 2, 3};
        std::vector<double> xsecs1 = {4, 3, 2, 1};
        int ZAID = 1001;

        CrossSectionBuilder xsbuilder( ZAID, energies1, xsecs1 );
        xsbuilder.setParticleType( photon );

        CrossSection xs = xsbuilder.construct();
        CHECK_EQUAL( photon, xs.getParticleType() );
    }

    TEST( CrossSection_read_write ) {
        std::vector<double> energies1 = {0, 1, 2, 3};
        std::vector<double> xsecs1 = {4, 3, 2, 1};
        int ZAID = 1001;

        CrossSectionBuilder xsbuilder( ZAID, energies1, xsecs1, photon, 1.33 );
        xsbuilder.setParticleType( photon );

        CrossSection xs = xsbuilder.construct();

        std::stringstream ss;
        xs.write(ss);

        CrossSectionBuilder xsbuilder2;
        xsbuilder2.read(ss);
        CrossSection xs2 = xsbuilder2.construct();

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

        CrossSectionBuilder xsbuilder( ZAID, energies1, xsecs1, photon, 1.33 );
        xsbuilder.setParticleType( photon );

        CrossSection xs = xsbuilder.construct();

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

        CrossSectionBuilder xsbuilder( ZAID, energies1, xsecs1, photon, 1.33 );
        xsbuilder.setParticleType( photon );

        CrossSection xs = xsbuilder.construct();

        CHECK_CLOSE( 4.0, xs.getTotalXS(0.1), 1e-5 );
        CHECK_CLOSE( 3.5, xs.getTotalXS(0.6), 1e-5 );
        CHECK_CLOSE( 2.5, xs.getTotalXS(1.5), 1e-5 );
        CHECK_CLOSE( 1.5, xs.getTotalXS(2.5), 1e-5 );
        CHECK_CLOSE( 1.0, xs.getTotalXS(3.5), 1e-5 );
    }

    TEST( read_neutron_file ) {

        CrossSectionBuilder xsbuilder;
        xsbuilder.read( std::string("MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin") );

        CrossSection xs = xsbuilder.construct();

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

    TEST( Test_U235_on_GPU ) {
        managed_vector<int> size_value;
        size_value.push_back( 0 );

        managed_vector<gpuFloatType_t> xs_value;
        xs_value.push_back( 0.0 );

        CrossSection* XS = new CrossSection;

        CrossSectionBuilder xsbuilder;
        xsbuilder.read( std::string("MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin") );

        *XS = xsbuilder.construct();
        deviceSynchronize();

#ifdef __CUDACC__
        kernelGetSize<<<1,1>>>( XS, size_value.data() );
        cudaDeviceSynchronize();
#else
        kernelGetSize( XS, size_value.data() );
#endif
        CHECK_EQUAL( 76525, size_value[0] );

        gpuFloatType_t energy = 2.0;
#ifdef __CUDACC__
        kernelTotalXS<<<1,1>>>( XS, xs_value.data(), energy );
        cudaDeviceSynchronize();
#else
        kernelTotalXS( XS, xs_value.data(), energy );
#endif
        CHECK_CLOSE( 7.14769f, xs_value[0], 1e-5);

        delete XS;
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

        double TotalXsec(double E, double T, unsigned i ) const {
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

        CrossSection XS = CrossSectionBuilder( xs ).construct();
        CHECK_EQUAL( 4, XS.size() );
        CHECK_EQUAL( 1001, XS.ZAID() );
        CHECK_CLOSE( 1.1, XS.getAWR(), 1e-5 );
    }

}

//TEST( read_neutron_file ) {
//
//    CrossSectionBuilder xsbuilder;
//    xsbuilder.read( std::string("MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin") );
//
//    CrossSection xs = xsbuilder.construct();
//
//    CHECK_EQUAL( 76525, xs.size() );
//    CHECK_CLOSE( 233.025, xs.getAWR(), 1e-3 );
//
//    gpuFloatType_t energy = 2.0;
//    double value = xs.getTotalXS(energy);
//    CHECK_CLOSE( 7.14769f, value, 1e-5);
//}

SUITE( CrossSection_speed_tester ) {

    template<typename EXECUTION_POLICY>
    void transformEnergyToXS( EXECUTION_POLICY policy,
                              managed_vector<gpuFloatType_t>& energies,
                              managed_vector<gpuFloatType_t>& results,
                              CrossSection* xs ) {
        auto calcXS = [=]  CUDA_CALLABLE_MEMBER (gpuFloatType_t E) { return xs->getTotalXS( E ); };

#ifdef __CUDACC__
        thrust::transform( policy, energies.data(), energies.data()+energies.size(),
                results.data(), calcXS );
#else
        std::transform( energies.data(), energies.data()+energies.size(),
                        results.data(), calcXS );
#endif

    }

    template<typename EXECUTION_POLICY>
    void transformEnergyToXSViaHash( EXECUTION_POLICY policy,
            managed_vector<gpuFloatType_t>& energies,
            managed_vector<gpuFloatType_t>& results,
            CrossSection* xs ) {
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

        CrossSectionBuilder xsbuilder;
        xsbuilder.read( std::string("MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin") );

        CrossSection* xs = new CrossSection;
        *xs = xsbuilder.construct();

        std::cout << "Debug: Starting U-235 neutron cross-section timing lookup test\n";
        cpuTimer timer;
        timer.start();

#ifdef __CUDACC__
        auto policy = thrust::host;
#else
        auto policy = false;
#endif

        transformEnergyToXS( policy, energies, xsecs, xs );
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

         CrossSectionBuilder xsbuilder;
         xsbuilder.read( std::string("MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin") );

         CrossSection* xs = new CrossSection;
         *xs = xsbuilder.construct();
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
         transformEnergyToXS( device_policy, energies, xsecs, xs );

         deviceSynchronize();
         timer.stop();
         std::cout << "Debug: Time to lookup " << nEnergies << " U-235 neutron cross-sections = " << timer.getTime() <<
                 " seconds\n";

#ifdef __CUDACC__
        auto host_policy = thrust::host;
#else
        auto host_policy = false;
#endif

         transformEnergyToXS( host_policy, energies, ref_xsecs, xs );

         for( unsigned i = 0; i < energies.size(); ++i ){
             gpuFloatType_t percent_diff = (ref_xsecs[i] - xsecs[i] ) / ref_xsecs[i];
             CHECK_CLOSE(0.0, percent_diff, 1e-5 );
         }

         delete xs;
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

         CrossSectionBuilder xsbuilder;
         xsbuilder.read( std::string("MonteRayTestFiles/92235-70c_MonteRayCrossSection.bin") );

         CrossSection* xs = new CrossSection;
         *xs = xsbuilder.construct();
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

         transformEnergyToXS( policy, energies, ref_xsecs, xs );

         std::cout << "Debug: Starting U-235 neutron cross-section timing lookup test on CPU via hash lookup\n";
         cpuTimer timer;
         timer.start();
         transformEnergyToXSViaHash( policy, energies, xsecs, xs );
         timer.stop();
         std::cout << "Debug: Time to perform hash lookup of " << nEnergies << " U-235 neutron cross-sections = " << timer.getTime() <<
                      " seconds\n";
         std::cout << "Debug: Complete\n";

         for( unsigned i = 0; i < energies.size(); ++i ){
             gpuFloatType_t percent_diff = (ref_xsecs[i] - xsecs[i] ) / ref_xsecs[i];
             CHECK_CLOSE(0.0, percent_diff, 1e-5 );
         }

         delete xs;
     }

}

} // end namespace
