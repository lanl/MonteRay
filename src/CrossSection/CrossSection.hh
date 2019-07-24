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

template<typename HASHFUNCTION = FasterHash >
class CrossSectionBuilder_t;

template< typename HASHFUNCTION = FasterHash >
class CrossSection_t : public Managed {
private:
    SimpleVector<gpuFloatType_t> energies;
    SimpleVector<gpuFloatType_t> totalXS;
    int id = 0;
    int ZA = 0;
    ParticleType_t ParticleType = neutron;
    gpuFloatType_t AWR = 0.0;

    CrossSectionHash_t<gpuFloatType_t, HASHFUNCTION>* hash = nullptr;

public:
    CrossSection_t(const CrossSection_t& rhs){
        id = rhs.id;
        ZA = rhs.ZA;
        AWR = rhs.AWR;
        ParticleType = rhs.ParticleType;
        energies.assign( rhs.energies.begin(), rhs.energies.end() );
        totalXS.assign( rhs.totalXS.begin(), rhs.totalXS.end() );
        build();
    };

    CrossSection_t& operator=(const CrossSection_t& rhs){
        id = rhs.id;
        ZA = rhs.ZA;
        AWR = rhs.AWR;
        ParticleType = rhs.ParticleType;

        size_t N = rhs.energies.size();
        energies.clear();
        totalXS.clear();
        energies.resize( N );
        totalXS.resize( N );
        for( size_t i = 0; i < rhs.energies.size(); ++i ) {
            energies[i] = rhs.energies[i];
            totalXS[i] = rhs.totalXS[i];
        }
        build();
        return *this;
    };

    CrossSection_t() = default;

    ~CrossSection_t(){
        if( hash ) delete hash;
    };

private:
    friend CrossSectionBuilder_t<HASHFUNCTION>;

    void build(){
        hash = new CrossSectionHash_t<gpuFloatType_t,HASHFUNCTION >( energies );
    };

public:
    void setID(int ID ){ id = ID; }
    constexpr int getID() const { return id; }
    constexpr int ZAID() const { return ZA; }
    constexpr size_t size() const { return energies.size(); }
    constexpr gpuFloatType_t getEnergy( size_t i ) const { return energies[i]; }
    constexpr gpuFloatType_t getTotalXSByIndex( size_t i ) const { return totalXS[i]; }
    constexpr gpuFloatType_t getAWR() const { return AWR; }
    constexpr ParticleType_t getParticleType() const { return ParticleType; }


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
        hash->getIndex(lower, upper, E);
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

private:
    template<typename S>
    void read( S& in ) {
        unsigned version;
        binaryIO::read(in, version );

        binaryIO::read(in, ParticleType );

        size_t numPoints;
        if( version > 0 ) {
            binaryIO::read(in, numPoints );
        } else {
            // version 0 wrote an int
            int N;
            binaryIO::read(in, N );
            numPoints = N;
        }
        binaryIO::read(in, AWR );

        if( version > 0 ) {
            binaryIO::read(in, ZA );
        }

        energies.resize( numPoints);
        totalXS.resize( numPoints);
        for( size_t i=0; i< this->size(); ++i ){
            binaryIO::read(in, energies[i] );
        }
        for( size_t i=0; i< this->size(); ++i ){
            binaryIO::read(in, totalXS[i] );
        }
    }

};

using CrossSection = CrossSection_t<>;

template< typename HASHFUNCTION >
class CrossSectionBuilder_t {
public:

    CrossSectionBuilder_t(){}

    template<typename T>
    CrossSectionBuilder_t( int ZAID,
                         std::vector<T> energies,
                         std::vector<T> xsec,
                         ParticleType_t type = neutron,
                         gpuFloatType_t AWR = 0.0
                       )
    {
        XS.ZA = ZAID;
        XS.energies.assign( energies.begin(), energies.end() );
        XS.totalXS.assign( xsec.begin(), xsec.end() );
        setParticleType( type );
        setAWR( AWR );
    }

    template<typename T>
    CrossSectionBuilder_t( const T& HostXS, double error = maxError, unsigned nBinsToVerify = nBinsToCheck ) {
        xsGrid_t xsGrid;

        auto xsFuncByIndex = [&] ( double E, size_t index ) {
            return getTotal<T>( HostXS, E, index );
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
            XS.energies.push_back( itr->first );
            XS.totalXS.push_back( itr->second );
        }
        setAWR( HostXS.getAWR() );
        XS.ZA = HostXS.getZAID();
    }

    void setAWR(gpuFloatType_t AWR) { XS.AWR = AWR; }
    void setParticleType( ParticleType_t ParticleType ) { XS.ParticleType = ParticleType; }

    ~CrossSectionBuilder_t(){};

    CrossSection_t<HASHFUNCTION> construct(){
        return XS;
    }

    // TODO: replace w/ readFromFile
    void read( const std::string& filename ) {
        std::ifstream infile;
        if( infile.is_open() ) {
            infile.close();
        }
        infile.open( filename.c_str(), std::ios::binary | std::ios::in);

        if( ! infile.is_open() ) {
            fprintf(stderr, "Debug:  CrossSectionBuilder::read -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
            exit(1);
        }
        assert( infile.good() );
        infile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
        read(infile);
        infile.close();
    }

    template<typename S>
    void read(S& in) {
        XS.read(in);
    }

    void setID(int newID){ XS.setID(newID); }

private:
    CrossSection_t<HASHFUNCTION> XS;
    static constexpr double maxError = 0.1;
    static const unsigned nBinsToCheck = 1000;
};

using CrossSectionBuilder = CrossSectionBuilder_t<>;

} /* namespace MonteRay */


#endif /* CROSSSECTION_HH_ */
