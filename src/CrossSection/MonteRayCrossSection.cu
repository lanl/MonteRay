#include "MonteRayCrossSection.hh"

#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "GPUErrorCheck.hh"
#include "MonteRay_binaryIO.hh"
#include "HashLookup.hh"
#include "MonteRayMemory.hh"

namespace MonteRay{

void ctor(struct MonteRayCrossSection* pXS, unsigned num) {
    if( num <=0 ) { num = 1; }

    pXS->id = -1;
    pXS->ParticleType = neutron;
    pXS->numPoints = num;
    pXS->AWR = 0.0;

    unsigned allocSize = sizeof(gpuFloatType_t)*num;
    pXS->energies  = (gpuFloatType_t*) malloc( allocSize);
    if (pXS->energies == 0) abort ();

    pXS->totalXS   = (gpuFloatType_t*) malloc( allocSize );
    if (pXS->totalXS == 0) abort ();

    for( unsigned i=0; i<num; ++i ){
        pXS->energies[i] = -1.0;
        pXS->totalXS[i] =  0.0;
    }
}

void dtor(struct MonteRayCrossSection* pXS) {
    if( pXS->energies != 0 ) {
        free(pXS->energies);
        pXS->energies = 0;
    }
    if( pXS->totalXS != 0 ) {
        free(pXS->totalXS);
        pXS->totalXS = 0;
    }
}


void cudaCtor(MonteRayCrossSection* ptr, unsigned num) {
#ifdef __CUDACC__
    ptr->numPoints = num;
    ptr->ParticleType = neutron;
    unsigned allocSize = sizeof( gpuFloatType_t ) * num;

    ptr->energies = (gpuFloatType_t*) MONTERAYDEVICEALLOC( allocSize, std::string("cudaCtor::ptr->energies") );
    ptr->totalXS = (gpuFloatType_t*) MONTERAYDEVICEALLOC( allocSize, std::string("cudaCtor::ptr->totalXS") );
#endif
}

void cudaCtor(MonteRayCrossSection* pCopy, MonteRayCrossSection* pOrig) {
#ifdef __CUDACC__
    unsigned num = pOrig->numPoints;
    cudaCtor( pCopy, num);

    pCopy->id = pOrig->id;
    pCopy->AWR = pOrig->AWR;
    pCopy->ParticleType = pOrig->ParticleType;

    unsigned allocSize = sizeof( gpuFloatType_t ) * pOrig->numPoints;

    CUDA_CHECK_RETURN( cudaMemcpy(pCopy->energies, pOrig->energies, allocSize, cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN( cudaMemcpy(pCopy->totalXS, pOrig->totalXS, allocSize, cudaMemcpyHostToDevice));
#endif
}

void cudaDtor(MonteRayCrossSection* ptr) {
#ifdef __CUDACC__
    MonteRayDeviceFree( ptr->energies );
    MonteRayDeviceFree( ptr->totalXS );
#endif
}

void copy(struct MonteRayCrossSection* pCopy, struct MonteRayCrossSection* pOrig ) {
    unsigned num = pOrig->numPoints;
    if( num <=0 ) { num = 1; }

    ctor( pCopy, num);
    pCopy->id = pOrig->id;
    pCopy->AWR = pOrig->AWR;
    pCopy->ParticleType = pOrig->ParticleType;
    for( unsigned i=0; i<num; ++i ){
        pCopy->energies[i] = pOrig->energies[i];
        pCopy->totalXS[i] =  pOrig->totalXS[i];
    }
}

CUDA_CALLABLE_MEMBER
ParticleType_t getParticleType(const struct MonteRayCrossSection* pXS) {
    return pXS->ParticleType;
}

CUDA_CALLABLE_MEMBER
void setParticleType( struct MonteRayCrossSection* pXS, ParticleType_t type) {
    pXS->ParticleType = type;
}

CUDA_CALLABLE_MEMBER
int getID(const struct MonteRayCrossSection* pXS) {
    return pXS->id;
}

CUDA_CALLABLE_MEMBER
void setID(struct MonteRayCrossSection* pXS, unsigned i) {
    if( pXS->id < 0 ) {
        pXS->id = i;
    }
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t getEnergy(const struct MonteRayCrossSection* pXS, unsigned i ) {
    return pXS->energies[i];
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXSByIndex(const struct MonteRayCrossSection* pXS, unsigned i ){
    return pXS->totalXS[i];
}

CUDA_CALLABLE_MEMBER
unsigned getIndex(const struct MonteRayCrossSection* pXS, gpuFloatType_t value ){
    // modified from http://en.cppreference.com/w/cpp/algorithm/upper_bound
    return getIndexBinary( pXS, 0, pXS->numPoints-1, value);
}

CUDA_CALLABLE_MEMBER
unsigned getIndexBinary(const struct MonteRayCrossSection* pXS, unsigned lower, unsigned upper, gpuFloatType_t value ){
    // modified from http://en.cppreference.com/w/cpp/algorithm/upper_bound
    unsigned it, step;
    unsigned first = lower;
    unsigned count = upper-lower+1;

    while (count > 0U) {
        it = first;
        step = count / 2;
        it += step;
        if(!(value < pXS->energies[it])) {
            first = ++it;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    if( first > 0 ) { --first; }
    return first;
}

CUDA_CALLABLE_MEMBER
unsigned getIndexLinear(const struct MonteRayCrossSection* pXS, unsigned lower, unsigned upper, gpuFloatType_t value ){

    for( unsigned i=lower+1; i < upper+1; ++i ){
        if( value < pXS->energies[ i ] ) {
            return i-1;
        }
    }
    if( value < pXS->energies[ lower ] ) { return lower; }
    return upper;
}

CUDA_CALLABLE_MEMBER
unsigned getIndex(const struct MonteRayCrossSection* pXS, const struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E ){
    unsigned isotope = MonteRay::getID(pXS);
    unsigned lowerBin = MonteRay::getLowerBoundbyIndex(pHash, isotope, hashBin);
    unsigned upperBin = MonteRay::getUpperBoundbyIndex(pHash, isotope, hashBin);

    if( upperBin-lowerBin+1 <= 8 ){
        return getIndexLinear( pXS, lowerBin, upperBin, E);
    } else {
        return getIndexBinary( pXS, lowerBin, upperBin, E);
    }

}


CUDA_CALLABLE_MEMBER
gpuFloatType_t getAWR(const struct MonteRayCrossSection* pXS) {
    return pXS->AWR;
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXSByIndex(const struct MonteRayCrossSection* pXS, unsigned i, gpuFloatType_t E ) {
    gpuFloatType_t lower =  pXS->totalXS[i];
    gpuFloatType_t upper =  pXS->totalXS[i+1];
    gpuFloatType_t deltaE = pXS->energies[i+1] - pXS->energies[i];

    gpuFloatType_t value = lower + (upper-lower) * (E - pXS->energies[i])/deltaE;

    return value;
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXS(const struct MonteRayCrossSection* pXS, gpuFloatType_t E ) {
    if( E >= pXS->energies[ pXS->numPoints-1] ) {
        gpuFloatType_t value = pXS->totalXS[ pXS->numPoints-1];
        return value;
    }

    if( E <= pXS->energies[ 0 ] ) {
        gpuFloatType_t value = pXS->totalXS[ 0 ];
        return value;
    }

    unsigned i = getIndex(pXS, E);
    return getTotalXSByIndex( pXS, i, E);
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t getTotalXS(const struct MonteRayCrossSection* pXS, const struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E ) {
    //	printf("Debug: MonteRayCrossSection::getTotalXS(const struct MonteRayCrossSection* pXS, const struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E )\n");
    if( E > pXS->energies[ pXS->numPoints-1] ) {
        return pXS->totalXS[ pXS->numPoints-1];
    }

    if( E < pXS->energies[ 0 ] ) {
        return pXS->totalXS[ 0 ];
    }

    unsigned i = getIndex(pXS, pHash, hashBin, E);
    return getTotalXSByIndex( pXS, i, E);
}

CUDA_CALLABLE_KERNEL  kernelGetTotalXS(const struct MonteRayCrossSection* pXS, const HashLookup* pHash, unsigned HashBin, gpuFloatType_t E, gpuFloatType_t* results){
    results[0] = getTotalXS(pXS, pHash, HashBin, E);
    return;
}


CUDA_CALLABLE_KERNEL  kernelGetTotalXS(const struct MonteRayCrossSection* pXS,  gpuFloatType_t E, gpuFloatType_t* results){
    results[0] = getTotalXS(pXS, E);
    return;
}

gpuFloatType_t
launchGetTotalXS( MonteRayCrossSectionHost* pXS, gpuFloatType_t energy){
    gpuFloatType_t result[1];

#ifdef __CUDACC__
    gpuFloatType_t* result_device;

    result_device = (gpuFloatType_t*) MONTERAYDEVICEALLOC( sizeof( gpuFloatType_t) * 1, std::string("launchGetTotalXS::result_device") );

    cudaEvent_t sync;
    cudaEventCreate(&sync);
    kernelGetTotalXS<<<1,1>>>( pXS->xs_device, energy, result_device);
    gpuErrchk( cudaPeekAtLastError() );
    cudaEventRecord(sync, 0);
    cudaEventSynchronize(sync);

    CUDA_CHECK_RETURN(cudaMemcpy(result, result_device, sizeof(gpuFloatType_t)*1, cudaMemcpyDeviceToHost));

    MonteRayDeviceFree( result_device );
#else
    kernelGetTotalXS( pXS->getPtr(), energy, result);
#endif
    return result[0];
}

MonteRayCrossSectionHost::MonteRayCrossSectionHost(unsigned num){
    xs = (struct MonteRayCrossSection*) MONTERAYHOSTALLOC( sizeof(struct MonteRayCrossSection), false, std::string("MonteRayCrossSectionHost::xs") );

    ctor(xs,num);

#ifdef __CUDACC__
    xs_device = (MonteRayCrossSection*) MONTERAYDEVICEALLOC( sizeof( MonteRayCrossSection), std::string("MonteRayCrossSectionHost::xs_device") );
#endif
}

MonteRayCrossSectionHost::~MonteRayCrossSectionHost(){
    dtor(xs);

    if( xs != 0 ) {
        MonteRayHostFree( xs, false );
        xs = 0;
    }

    if( cudaCopyMade ) {
        cudaDtor( temp );
        delete temp;
    }
#ifdef __CUDACC__
    MonteRayDeviceFree( xs_device );
#endif
}

gpuFloatType_t MonteRayCrossSectionHost::getTotalXS( const struct HashLookup* pHash, unsigned hashBin, gpuFloatType_t E ) const {
    return MonteRay::getTotalXS(xs, pHash, hashBin, E);
}

gpuFloatType_t MonteRayCrossSectionHost::getTotalXSByHashIndex(const struct HashLookup* pHash, unsigned i, gpuFloatType_t E) const {
    return MonteRay::getTotalXS(xs, pHash, i, E);
}

unsigned MonteRayCrossSectionHost::getIndex( const HashLookupHost* pHost, unsigned hashBin, gpuFloatType_t e ) const {
    return MonteRay::getIndex( xs, pHost->getPtr(), hashBin, e);
}


void MonteRayCrossSectionHost::copyToGPU(void) {
#ifdef __CUDACC__
    cudaCopyMade = true;

    if( temp ) {
        cudaDtor(temp);
        delete temp;
    }
    temp = new MonteRayCrossSection;
    cudaCtor(temp, xs );
    CUDA_CHECK_RETURN( cudaMemcpy(xs_device, temp, sizeof( MonteRayCrossSection ), cudaMemcpyHostToDevice));
#endif
}

void MonteRayCrossSectionHost::load(struct MonteRayCrossSection* ptrXS ) {
    unsigned num = ptrXS->numPoints;
    dtor( xs );
    ctor( xs, num );

    setAWR( ptrXS->AWR );

    for( unsigned i=0; i<num; ++i ){
        gpuFloatType_t energy = ptrXS->energies[i];
        gpuFloatType_t totalXS = ptrXS->totalXS[i];
        xs->energies[i] = energy;
        xs->totalXS[i] = totalXS;
    }
}

void MonteRayCrossSectionHost::write(std::ostream& outf) const{
    unsigned CrossSectionFileVersion = 0;
    binaryIO::write(outf, CrossSectionFileVersion );

    binaryIO::write(outf, xs->ParticleType );
    binaryIO::write(outf, xs->numPoints );
    binaryIO::write(outf, xs->AWR );
    for( unsigned i=0; i<xs->numPoints; ++i ){
        binaryIO::write(outf, xs->energies[i] );
    }
    for( unsigned i=0; i<xs->numPoints; ++i ){
        binaryIO::write(outf, xs->totalXS[i] );
    }
}

void MonteRayCrossSectionHost::read(std::istream& infile) {
    const bool debug = false;
    unsigned CrossSectionFileVersion;
    binaryIO::read(infile, CrossSectionFileVersion);
    if( debug ) printf("Debug: MonteRayCrossSectionHost::read -- CrossSectionFileVersion=%d\n", CrossSectionFileVersion);

    ParticleType_t particleType;
    binaryIO::read(infile, particleType );

    unsigned num;
    binaryIO::read(infile, num);
    dtor( xs );
    ctor( xs, num );

    xs->ParticleType = particleType;
    if( debug ) printf("Debug: MonteRayCrossSectionHost::read -- ParticleType=%d\n", xs->ParticleType);

    binaryIO::read(infile, xs->AWR );
    for( unsigned i=0; i<num; ++i ){
        binaryIO::read(infile, xs->energies[i] );
    }

    for( unsigned i=0; i<num; ++i ){
        binaryIO::read(infile, xs->totalXS[i] );
    }
}


void MonteRayCrossSectionHost::write( const std::string& filename ) {
    std::ofstream outfile;

    outfile.open( filename.c_str(), std::ios::binary | std::ios::out);
    if( ! outfile.is_open() ) {
        fprintf(stderr, "MonteRayCrossSectionHost::write -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( outfile.good() );
    outfile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    write( outfile );
    outfile.close();
}

void MonteRayCrossSectionHost::read( const std::string& filename ) {
    std::ifstream infile;
    if( infile.is_open() ) {
        infile.close();
    }
    infile.open( filename.c_str(), std::ios::binary | std::ios::in);

    if( ! infile.is_open() ) {
        fprintf(stderr, "Debug:  MonteRayCrossSectionHost::read -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( infile.good() );
    infile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    read(infile);
    infile.close();
}


void
MonteRayCrossSectionHost::thinGrid(const totalXSFunct_t& xsFunc, linearGrid_t& linearGrid, double max_error) {
    // thin grid
    bool done;
    do {
        done = true;
        unsigned i = 0;
        for( auto previous_itr = linearGrid.begin(); previous_itr != linearGrid.end(); ++previous_itr) {
            auto itr = previous_itr; ++itr;
            if( itr == linearGrid.end() ) break;
            auto next_itr = itr; ++next_itr;
            if( next_itr == linearGrid.end() ) break;

            // check log mid-point
            double energy1 = previous_itr->first;
            double energy2 = next_itr->first;
            double energy = itr->first;

            // calculated interpolatedXS
            double lower =  previous_itr->second;
            double upper =  next_itr->second;
            double deltaE = energy2 - energy1;
            double interpolatedXS = lower + (upper-lower) * (energy - energy1)/deltaE;

            // check difference with real xs
            double totalXS = xsFunc( energy );
            double percentDiff = std::abs(totalXS - interpolatedXS ) * 100.0 / totalXS;
            //            printf( "Debug: i=%d  E=%f, interp=%f, real=%f diff=%f \n", i, energy, interpolatedXS, totalXS, percentDiff);
            if( percentDiff < max_error * 0.5 ) {
                linearGrid.erase(itr);
                done = false;
                break;
            }
            ++i;
        }
    } while( !done );
}

void
MonteRayCrossSectionHost::addPointsToGrid(const totalXSFunct_t& xsFunc, linearGrid_t& linearGrid, double max_error) const {
    bool done;
    // linearize
    do {
        done = true;
        for( auto previous_itr = linearGrid.begin(); previous_itr != linearGrid.end(); ++previous_itr) {
            auto itr = previous_itr; ++itr;
            if( itr == linearGrid.end() ) break;

            // check log mid-point
            double energy1 = previous_itr->first;
            double energy2 = itr->first;
            double deltaE = energy2 - energy1;

            if( deltaE > 1e-6 ) {
                // don't add points finer than 1e-6
                double energy = std::exp(( std::log(energy2) - std::log(energy1) )*0.5 + std::log(energy1));

                // calculated interpolatedXS
                double lower =  previous_itr->second;
                double upper =  itr->second;

                double interpolatedXS = lower + (upper-lower) * (energy - energy1)/deltaE;

                // check difference with real xs
                double totalXS = xsFunc( energy );
                double percentDiff = std::abs(totalXS - interpolatedXS ) * 100.0 / totalXS;

                if( percentDiff > max_error ) {
                    linearGrid.insert(itr, std::make_pair(energy, totalXS));
                    done = false;
                }
            }
        }

    } while ( !done );
}

bool
MonteRayCrossSectionHost::checkGrid(const totalXSFunct_t& xsFunc, linearGrid_t& linearGrid, double max_error, unsigned nIntermediateBins) const{
    const bool debug = false;

    if( debug ) printf( "Debug: MonteRayCrossSectionHost::createLinearGrid - checking linearization\n");

    // check linearization
    bool done = true;
    do {
        done = true;
        auto start_itr = linearGrid.begin();
        for( auto previous_itr = start_itr; previous_itr != linearGrid.end(); ++previous_itr) {
            auto itr = previous_itr; ++itr;
            if( itr == linearGrid.end() ) break;

            // check log mid-point
            double energy1 = previous_itr->first;
            double energy2 = itr->first;
            double deltaE = energy2 - energy1;

            double lower =  previous_itr->second;
            double upper =  itr->second;

            // no need to go below 1-eV for photon data
            if( std::abs( deltaE ) > 1e-6 ) {
                nIntermediateBins = std::min( unsigned( deltaE / 1e-6 ), nIntermediateBins ) ;
            } else {
                nIntermediateBins = 0;
            }

            for( auto j=0; j<nIntermediateBins; ++j) {
                double energy = energy1 + (deltaE*j)/nIntermediateBins;
                // calculated interpolatedXS
                double interpolatedXS = lower + (upper-lower) * (energy - energy1)/deltaE;
                double totalXS = xsFunc( energy );
                double percentDiff = std::abs(totalXS - interpolatedXS ) * 100.0 / totalXS;

                if( percentDiff > max_error ) {
                    if( debug ) {
                        printf( "Debug: MonteRayCrossSectionHost::createLinearGrid - linearization failed for E=%.10f, real XS=%f, interpolated XS=%f, percent diff=%f\n",
                                energy, totalXS, interpolatedXS, percentDiff );

                    }
                    start_itr = linearGrid.insert(itr, std::make_pair(energy, totalXS));
                    done = false;
                    break;
                }

                if( debug ) {
                    printf( "Debug: MonteRayCrossSectionHost::createLinearGrid - linearization passed for E=%.10f, real XS=%f, interpolated XS=%f, percent diff=%f\n",
                            energy, totalXS, interpolatedXS, percentDiff );
                }
            }

        }
    } while ( !done );
    return true;

}

}
