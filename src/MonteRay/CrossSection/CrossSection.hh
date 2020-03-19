#ifndef CROSSSECTION_HH_
#define CROSSSECTION_HH_

#include <fstream>

#include "MonteRayTypes.hh"
#include "MonteRayConstants.hh"
#include "ManagedAllocator.hh"
#include "SimpleVector.hh"
#include "CrossSectionUtilities.hh"
#include "MonteRay_binaryIO.hh"
#include "CrossSectionHash.hh"

#include "BinarySearch.hh"


namespace MonteRay {

template<typename HashFunction = FasterHash >
class CrossSectionBuilder_t;

template< typename HashFunction = FasterHash >
class CrossSection_t : public Managed {
private:
    SimpleVector<gpuFloatType_t> energies;
    SimpleVector<gpuFloatType_t> totalXS;
    int ZA = 0;
    ParticleType_t particleType = neutron;
    gpuFloatType_t AWR = 0.0;
    CrossSectionHash_t<gpuFloatType_t, HashFunction> hash;

public:
    CrossSection_t( SimpleVector<gpuFloatType_t> energiesIn, 
                    SimpleVector<gpuFloatType_t> totalXSIn, 
                    int ZAIn, 
                    ParticleType_t particleTypeIn, 
                    gpuFloatType_t AWRIn, 
                    CrossSectionHash_t<gpuFloatType_t, HashFunction> hashIn
                  ):
      energies(std::move(energiesIn)), 
      totalXS(std::move(totalXSIn)), 
      ZA(ZAIn), 
      particleType(particleTypeIn), 
      AWR(AWRIn), 
      hash(std::move(hashIn))
    {}

    CrossSection_t( int ZAIn, 
                    SimpleVector<gpuFloatType_t> energiesIn, 
                    SimpleVector<gpuFloatType_t> totalXSIn, 
                    ParticleType_t particleTypeIn, 
                    gpuFloatType_t AWRIn
                  ): CrossSection_t(std::move(energiesIn), std::move(totalXSIn), ZAIn, particleTypeIn, AWRIn,
                                    CrossSectionHash_t<gpuFloatType_t, HashFunction>( energies ))
    {}

    template <typename OtherContainer>
    CrossSection_t( int ZAIn, 
                    OtherContainer  energiesIn, 
                    OtherContainer  totalXSIn, 
                    ParticleType_t particleTypeIn = neutron,
                    gpuFloatType_t AWRIn = 0.0
                  )
    {
      energies.assign(energiesIn.begin(), energiesIn.end() );
      totalXS.assign( totalXSIn.begin(), totalXSIn.end() );
      ZA = ZAIn;
      particleType = particleTypeIn;
      AWR = AWR;
      hash = CrossSectionHash_t<gpuFloatType_t, HashFunction>( energies );
    }

    CrossSection_t() = default;

private:
    friend CrossSectionBuilder_t<HashFunction>;

public:
    constexpr int ZAID() const { return ZA; }
    constexpr size_t size() const { return energies.size(); }
    constexpr gpuFloatType_t getEnergy( size_t i ) const { return energies[i]; }
    constexpr gpuFloatType_t getTotalXSByIndex( size_t i ) const { return totalXS[i]; }
    constexpr gpuFloatType_t getAWR() const { return AWR; }
    constexpr ParticleType_t getParticleType() const { return particleType; }


    constexpr size_t getIndex( gpuFloatType_t E ) const {
      return LowerBoundIndex(energies.data(), 0U, this->size(), E );
    }

    // TODO: make this private?  or move it into getTotalXS
    constexpr gpuFloatType_t getTotalXS(size_t i, gpuFloatType_t E ) const {
        gpuFloatType_t lowerXS =  getTotalXSByIndex(i);
        gpuFloatType_t lowerEnergy =  getEnergy(i);

        // off the top end of the table
        if( i >= this->size()-1 ) {
            return lowerXS;
        }

        // off the bottom end of the table
        if( E < lowerEnergy ) {
            return lowerXS;
        }

        gpuFloatType_t upperXS =  getTotalXSByIndex(i+1);
        gpuFloatType_t upperEnergy =  getEnergy(i+1);

        // interpolate
        gpuFloatType_t value = lowerXS + (upperXS-lowerXS) * (E - lowerEnergy)/(upperEnergy-lowerEnergy);

        return value;
    }

    // TODO: combine getTotalXS and getTotalXSViaHash
    constexpr gpuFloatType_t getTotalXS(gpuFloatType_t E ) const {

        return getTotalXS( getIndex(E), E);
    }

    // TODO: combine getTotalXS and getTotalXSViaHash
    constexpr gpuFloatType_t getTotalXSviaHash(gpuFloatType_t E ) const {
        size_t lower, upper;
        hash.getIndex(lower, upper, E);
        size_t index = LowerBoundIndex( energies.data(), lower, upper, E);
        return getTotalXS( index, E);
    }

    // TODO: replace with writeToFile
    void write( const std::string& filename ) const {
        std::ofstream outfile;

        outfile.open( filename.c_str(), std::ios::binary | std::ios::out);
        if( ! outfile.is_open() ) {
            fprintf(stderr, "CrossSection::write -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
            exit(1);
        }
        assert( outfile.good() );
        outfile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
        write( outfile );
        outfile.close();
    }

    void write( std::ostream& out) const {
        unsigned CrossSectionFileVersion = 1;
        binaryIO::write(out, CrossSectionFileVersion );

        binaryIO::write(out, getParticleType() );
        binaryIO::write(out, this->size() );
        binaryIO::write(out, getAWR() );
        binaryIO::write(out, ZAID() ); // version 1
        for( size_t i=0; i< this->size(); ++i ){
            binaryIO::write(out, getEnergy(i) );
        }
        for( size_t i=0; i< this->size(); ++i ){
            binaryIO::write(out, getTotalXSByIndex(i) );
        }
    }

    template<typename S>
    static auto read( S& in ) {
        unsigned version;
        binaryIO::read(in, version );

        ParticleType_t particleType;
        binaryIO::read(in, particleType );

        size_t numPoints;
        if( version > 0 ) {
            binaryIO::read(in, numPoints );
        } else {
            // version 0 wrote int
            unsigned N;
            binaryIO::read(in, N );
            numPoints = N;
        }
        gpuFloatType_t AWR = 0.0;
        binaryIO::read(in, AWR );

        int ZA = 0;
        if( version > 0 ) {
            binaryIO::read(in, ZA );
        }

        SimpleVector<gpuFloatType_t> energies;
        SimpleVector<gpuFloatType_t> totalXS;
        energies.resize( numPoints);
        totalXS.resize( numPoints);
        for( size_t i=0; i< numPoints; ++i ){
            binaryIO::read(in, energies[i] );
        }
        for( size_t i=0; i< numPoints; ++i ){
            binaryIO::read(in, totalXS[i] );
        }

        CrossSectionHash_t<gpuFloatType_t, HashFunction> hash( energies );
        return CrossSection_t( std::move(energies), std::move(totalXS), ZA, particleType, AWR, std::move(hash));

    }

};

/* template<typename T> */
/* using CrossSectionBuilder = CrossSection_t<T>::Builder */

using CrossSection = CrossSection_t<>;

template< typename HashFunction >
class CrossSectionBuilder_t {
public:

    CrossSectionBuilder_t(){}

    SimpleVector<gpuFloatType_t> b_energies;
    SimpleVector<gpuFloatType_t> b_totalXS;
    int b_ZA = 0;
    ParticleType_t b_particleType = neutron;
    gpuFloatType_t b_AWR = 0.0;
    CrossSectionHash_t<gpuFloatType_t, HashFunction> b_hash;

    template<typename Container>
    CrossSectionBuilder_t( int ZAID,
                           Container&& energies,
                           Container&& xsec,
                           ParticleType_t type = neutron,
                           gpuFloatType_t AWR = 0.0
                         )
    {
      b_energies.assign(energies.begin(), energies.end() );
      b_totalXS.assign( xsec.begin(), xsec.end() );
      b_ZA = ZAID;
      b_particleType = type;
      b_AWR = AWR;
    }

    template<typename T>
    CrossSectionBuilder_t( const T& HostXS, double error = maxError, unsigned nBinsToVerify = nBinsToCheck ) {
        xsGrid_t xsGrid;

        auto xsFuncByIndex = [&] ( size_t index ) {
            return getTotal<T>( HostXS, index );
        };
        auto xsFunc = [&] ( double E ) {
            return getTotal<T>( HostXS, E );
        };

        if( HostXS.getType() == std::string("neutron") ) {
            setParticleType( neutron );
            auto toEnergyFunc = [&] (double E) -> double {
                return E;
            };

            xsGrid = createXSGrid<T,xsGrid_t>(HostXS, toEnergyFunc, xsFuncByIndex);
        }
        if( HostXS.getType() == std::string("photon") ) {
            setParticleType( photon );
            auto toEnergyFunc = [&] (double E) -> double {
                return std::exp(E);
            };

            linearGrid_t linearGrid = createXSGrid<T,linearGrid_t>( HostXS, toEnergyFunc, xsFuncByIndex );
            thinGrid( xsFunc, linearGrid, error );

            addPointsToGrid( xsFunc, linearGrid, error );
            checkGrid( xsFunc, linearGrid, error, nBinsToVerify );

            xsGrid.reserve( linearGrid.size() );
            std::copy( std::begin(linearGrid), std::end(linearGrid), std::back_inserter(xsGrid));
        }

        for( auto itr = xsGrid.begin(); itr != xsGrid.end(); ++itr) {
            b_energies.push_back( itr->first );
            b_totalXS.push_back( itr->second );
        }
        b_AWR = HostXS.getAWR();
        b_ZA = HostXS.getZAID();
    }

    void setZAID( int ZA ) { b_ZA = ZA; }
    void setAWR(gpuFloatType_t AWR) { b_AWR = AWR; }
    void setParticleType( ParticleType_t particleTypeIn ) { b_particleType = particleTypeIn; }


    CrossSection_t<HashFunction> build() {
      b_hash = CrossSectionHash_t<gpuFloatType_t, HashFunction>( b_energies );
      return {std::move(b_energies), std::move(b_totalXS), b_ZA, b_particleType, b_AWR, std::move(b_hash)};
    }

    auto construct(){
      return this->build();
    }

    template<typename S>
    void read( S& in ) {
        unsigned version;
        binaryIO::read(in, version );

        binaryIO::read(in, b_particleType );

        size_t numPoints;
        if( version > 0 ) {
            binaryIO::read(in, numPoints );
        } else {
            // version 0 wrote int
            unsigned N;
            binaryIO::read(in, N );
            numPoints = N;
        }
        binaryIO::read(in, b_AWR );

        if( version > 0 ) {
            binaryIO::read(in, b_ZA );
        }

        b_energies.resize( numPoints);
        b_totalXS.resize( numPoints);
        for( size_t i=0; i< numPoints; ++i ){
            binaryIO::read(in, b_energies[i] );
        }
        for( size_t i=0; i< numPoints; ++i ){
            binaryIO::read(in, b_totalXS[i] );
        }
    }


private:
    CrossSection_t<HashFunction> XS;
    static constexpr double maxError = 0.1;
    static const unsigned nBinsToCheck = 1000;
};

using CrossSectionBuilder = CrossSectionBuilder_t<>;

} /* namespace MonteRay */


#endif /* CROSSSECTION_HH_ */
