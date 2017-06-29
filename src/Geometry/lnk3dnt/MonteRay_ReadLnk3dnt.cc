#include "MonteRay_ReadLnk3dnt.hh"

// #include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <map>

#include "MonteRay_BinaryReadFcns.hh"

#include <stdexcept>


using namespace std;

namespace MonteRay  {

bool
MonteRay_exists( const std::string& filename) {
    if (FILE *file = fopen(filename.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }

}

string
MonteRay_checkForLnk3dnt(const std::string& filename) {
    if( MonteRay_exists(filename) ) {
    	return filename;
    } else {
    	stringstream msg;
    	msg     << "ERROR: Unable to locate geometry file: ** " <<filename<<"\n"
    			<< "Generated from : "<< __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_checkForLnk3dnt" <<endl;
    	throw runtime_error( msg.str() );
    }

}

MonteRay_ReadLnk3dnt::MonteRay_ReadLnk3dnt( const std::string& filename )
{
    try {
        std::string FoundFile = MonteRay_checkForLnk3dnt(filename);

        file.open( FoundFile );
        file.setMark( "Begin" );
        ReadGeomData();

        // Mark the current position as the start of the MatIds file.
        file.setMark( "beginMatInfo");

    } catch( const std::exception& err ) {
        stringstream msg;
        msg << err.what();
        msg << "Called from : "<< __FILE__<<__LINE__<<"] : "<< "MonteRay_ReadLnk3dnt::MonteRay_ReadLnk3dnt" <<endl;
        throw std::runtime_error( msg.str() );
    }
    TotalNumCells = calcNumTotalCells();
}


MonteRay_ReadLnk3dnt::Partisn_int
MonteRay_ReadLnk3dnt::getMaxMatID( void ) const
{
    if( UniqueMatIDs.size() > 0 )
        return *max_element(UniqueMatIDs.begin(), UniqueMatIDs.end() );
    else
        return 0;
}
/// ReadGeomData: Handles a BinaryReader object and reads in all the Meta data in a LNK3DNT file. \n
///   Metadata describes the type and length of subsequent data arrays, and therefore metadata must \n
///   be read first.  Examples include file version, type of geometry, number of zones. \n

/// \param file is a BinaryReader object that holds the data from a LNK3DNT.
/// \param
void
MonteRay_ReadLnk3dnt::ReadGeomData()
{
    stringstream errorMsg( "Invalid link file:  ");

    Partisn_int EmptyInt;

    {std::string junkString;
    file.read( junkString, CharLength );
    file.read( junkString, CharLength );
    file.read( junkString, CharLength );
    }

    file.testByteOrder( FileVersion );
    if( FileVersion < 0 || FileVersion > 100 )
        file.toggleByteSwapping();

    file.read( FileVersion );
    if( FileVersion < 0 || FileVersion > 5 ) {
        errorMsg << "File version unrecognized.  File version >> "<<FileVersion<<" << "<<endl;
        throw std::runtime_error( errorMsg.str() );
    }

    std::map<int, DIMEN > GeoToDim;
    GeoToDim[ SPHERICAL ] = OneD;
    GeoToDim[ RZ ]        = TwoD;
    GeoToDim[ XYZ ]       = ThreeD;
    GeoToDim[ RZT ]       = ThreeD;

    file.read( GeometrySpec );
    if( GeoToDim.find(GeometrySpec) == GeoToDim.end() ) {
        errorMsg << "Geometry specification unrecognized.  Geometry = >> "<<GeometrySpec<<" <<"<<endl;
        throw std::runtime_error( errorMsg.str() );
    }

    Dim = GeoToDim[ GeometrySpec ];

    file.read( NZones );
    if( NZones < 0 || NZones > 100000000 ) {
        errorMsg << "Invalid number of zones specified.  NZones = >> "<<NZones<<" <<"<<endl;
        throw std::runtime_error( errorMsg.str() );
    }

    for( unsigned i=0; i<2; ++i )
        file.read( EmptyInt );

    ///\brief MaterialRegion is the toolkit's interpretation of 'coarse' cells as mentioned in the Partisn manual.
    ///
    /// \detail MaterialRegion is the toolkit's interpretation of 'coarse' cells as mentioned in the Partisn manual.
    /// It contains the number of regions along a dimension.  Within each of these regions, the material and
    /// mesh properties are constant.  Each region is defined by the material it contains, the number of cells
    /// into which it has been divided and the spatial bounds of its relevance.  Naturally, the boundaries of
    /// the regions must all touch exactly with their neighbors.  The sum of the number of cells across all regions
    /// will equal the number of 'fine' cells.  While the solution grid is specified on the fine cell level,
    /// the vertices listed on the lnk3dnt are only specified at the region boundaries.  As such, they must
    /// be converted to each region's fine mesh level to supply the fine mesh vertices.\n
    ///
    /// Example:<pre>
    /// ||                              Mat 1                              |       Mat 2       ||
    /// ||-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|---|---|---|---|---||
    /// ||  5     5     5     5     5     5     5     5     5     5     5  | 3   3   3   3   3 ||
    ///  0                                                                 55                  70</pre>
    /// The illustration shows an axis that is comprised of 2 regions.  The first region is bounded by 0 and 55.
    /// It contains material 1, and is divided into 11 'fine' cells each with a delta of 5 units.
    /// Region 2 contains material 2 and is bounded by 55 and 70.  It contains 5 'fine' cells with a common
    /// delta of 3 units.  This problem would have a total of 16 cells in its solution along this axis.
    /// for the x-dimen the partisn input would be :
    /// in Block 1:  im=2 it=16
    /// in Block 2:  xmesh=0.0 55.0 70.0  xints = 11 5 zones=1 2
    struct MaterialRegion {
        typedef Partisn_int   int_t;
        typedef double        real_t;
        int_t           NRegions;

        vector< real_t > Vertices;
        vector< int_t  > NCells;
        vector< real_t > combinedVertices( void ) {
            verifyAscending( Vertices );
            vector< real_t > combined;
            combined.push_back( Vertices[ 0 ] );
            for( int_t currentRegion = 0; currentRegion < NRegions; ++currentRegion ) {

                // extract the parameters describing the spatial properties of the current region
                real_t start  = Vertices[ currentRegion ];
                real_t finish = Vertices[ currentRegion + 1 ];
                real_t delta = ( finish - start ) / NCells[ currentRegion ];

                // start this loop with j=1 because the first vertex of the current region has already
                // been added since it is the same (or should be!) as the last vertex from the previous region.
                for( int_t j=1; j<NCells[ currentRegion ]; ++j )
                    combined.push_back( start + j * delta );
                combined.push_back( finish );
            }
            verifyAscending( combined );
            return combined;
        }
        void verifyAscending( const std::vector<real_t>& vertices ) const {
            for( unsigned i=1; i<vertices.size(); ++i ) {
                if( vertices[ i ] > vertices[ i - 1 ] ) continue;
                throw std::runtime_error( "Failure in MonteRay_ReadLnk3dnt.  Vertices are not strictly increasing." );
            }
        }
    };
    MaterialRegion RegionInfo[ MAXDIM ];

    for( unsigned d=0; d<MAXDIM; ++d ) {
        file.read( RegionInfo[ d ].NRegions );
    }

    for( unsigned d=0; d<MAXDIM; ++d )
        file.read( nCells[ d ] );

    for( unsigned i=0; i<13; ++i )
        file.read( EmptyInt );

    file.read( MaxNMatsInCell );

    for( unsigned i=0; i<2; ++i )
        file.read( EmptyInt );

    file.read( BlockLevels );

    // Read in the material region information for each dimension
    // Vertices for each region
    for( int d=0; d<Dim; ++d ) {
        file.read( RegionInfo[ d ].Vertices, RegionInfo[ d ].NRegions + 1 );
    }
    // Number of equally sized cells within each region
    for( int d=0; d<Dim; ++d ) {
        file.read( RegionInfo[ d ].NCells,   RegionInfo[ d ].NRegions );
    }

    // Combine the vertices from the regions along each dimension into a single vertex array
    for( int d=0; d<Dim; ++d ) {
        VertexPositions[ d ] = RegionInfo[ d ].combinedVertices();
        if( static_cast<int>( VertexPositions[ d ].size() ) != nCells[ d ]+1 ) {
            throw std::runtime_error( "Problem composing vertex information from lnk3dnt file." );
        }
    }

}


std::string
MonteRay_ReadLnk3dnt::getGeometryString( void ) const{

    std::map<int, std::string > GeoToString;
    GeoToString[ SPHERICAL ] = "Spherical";
    GeoToString[ RZ ]        = "RZ";
    GeoToString[ XYZ ]       = "XYZ";
    GeoToString[ RZT ]       = "RZT";

    return GeoToString[ GeometrySpec ];
}

void
MonteRay_ReadLnk3dnt::ReadMatData( void )
{
    // Ensure the file is at the beginning of the Material/Density region (i.e. past the meta data)
    file.useMark( "beginMatInfo" );

    //   if( FileVersion == 5 ) {
    if( FileVersion != 4 ) {
        // Add throw for AMR mesh in lnk3dnt
        std::stringstream msg;
        msg  << "ERROR:  LNK3DNT Reader will not read file version "<< FileVersion << std::endl;
        throw runtime_error( msg.str() );
    }

    unsigned NMeshCells = 1;
    for(int d=0; d<Dim;++d)
        NMeshCells *= nCells[ d ];

    unsigned NEntries = NMeshCells * MaxNMatsInCell;

    file.read( MatIDs,  NEntries );
    file.read( Density, NEntries );

    // Construct a list of the unique material ids that were read
    UniqueMatIDs.insert( MatIDs.begin(), MatIDs.end() );

    if ( *min_element(UniqueMatIDs.begin(), UniqueMatIDs.end()) < 1 ) {
        std::stringstream msg;
        msg  << "ERROR:  LNK3DNT File contains a negative or 0 MatId:"<< *min_element(MatIDs.begin(), MatIDs.end()) << std::endl;
        throw runtime_error( msg.str() );
    }
    if ( *min_element(Density.begin(), Density.end()) < 0.0 ) {
        std::stringstream msg;
        msg  << "ERROR: LNK3DNT File contains a negative density:"<< *min_element(Density.begin(), Density.end())  << std::endl;
        throw runtime_error( msg.str() );
    }
}

void
MonteRay_ReadLnk3dnt::Print(ostream& out) const {
    int index = 0;
    for( int n=0; n<MaxNMatsInCell; ++n ) {
        out << "MatID[ "<<n<<" ] ****** "<<endl;
        for( int k=0; k<nCells[2]; ++k ) {
            if( Dim == ThreeD )
                out << " z[ "<<setw(3)<<k<<" ] : "<<setprecision(12)<<VertexPositions[2][ k ]<<endl<<endl;
            for( int j=0; j<nCells[1]; ++j ) {
                for( int i=0; i<nCells[0]; ++i ) {
                    out <<setw(4)<< MatIDs[ index++ ];
                }
                out << endl;
            }
            out << endl;
        }
    }

    index = 0;
    for( int n=0; n<MaxNMatsInCell; ++n ) {
        out << "Density[ "<<n<<" ] ****** "<<endl;
        for( int k=0; k<nCells[2]; ++k ) {
            if( Dim == ThreeD )
                out << " z[ "<<setw(3)<<k<<" ] : "<<setprecision(12)<<VertexPositions[2][ k ]<<endl<<endl;
            for( int i=0; i<nCells[0]; ++i ) {
                out << setw(15) << setprecision(12)<< VertexPositions[ 0 ][ i ];
            }
            out << endl;
            for( int j=0; j<nCells[1]; ++j ) {
                if( Dim == TwoD )
                    out << setw(15)<<VertexPositions[ 1 ][ j ];
                for( int i=0; i<nCells[0]; ++i ) {
                    out << setw(15) << Density[ index++ ];
                }
                out << endl;
            }
        }
    }

    out << "---------------------------" << endl << endl;

    for( int d=0; d<Dim; ++d ) {
        out << "Vertex[ "<<d<<" ] : { ";
        for( vector<double>::const_iterator itr=VertexPositions[d].begin(); itr != VertexPositions[d].end(); ++itr )
            out << *itr << "  ";
        out << " } " <<endl;
    }

    out << endl;

}

//TRA - I used this a bit more than the other one. Left the other to keep MCNP compare runs diffing.
//void
//MonteRay_ReadLnk3dnt::Print(ostream& out) const {
//
//    out << "---------------------------" << endl << endl;
//
//    for( int d=0; d<Dim; ++d ) {
//        out << "Vertex[ "<<d<<" ] : { ";
//        for( vector<double>::const_iterator itr=VertexPositions[d].begin(); itr != VertexPositions[d].end(); ++itr )
//            out << *itr << "  ";
//        out << " } " <<endl;
//    }
//    out << endl;
//    out << "nCells for d=1,2,3 -> " << nCells[0] << " : " << nCells[1] << " : " << nCells[2] << endl;
//    out << "---------------------------" << endl << endl;
//
//    int index = 0;
//    out << "MaxNMatsInCell = " << MaxNMatsInCell << endl;
//    for( int n=0; n<MaxNMatsInCell; ++n ) {
//        out << "MatID[ "<<n<<" ] ****** "<<endl;
//        for( int k=0; k<nCells[2]; ++k ) {
//            for( int j=0; j<nCells[1]; ++j ) {
//                for( int i=0; i<nCells[0]; ++i ) {
//                    out <<setw(4)<< MatIDs[ index++ ];
//                }
//                out << endl;
//            }
//            out << endl;
//        }
//    }
//
//    index = 0;
//    for( int n=0; n<MaxNMatsInCell; ++n ) {
//        out << "Density[ "<<n<<" ] ****** "<<endl;
//        for( int k=0; k<nCells[2]; ++k ) {
//            for( int j=0; j<nCells[1]; ++j ) {
//                for( int i=0; i<nCells[0]; ++i ) {
//                    out << setw(15) << Density[ index++ ];
//                }
//                out << endl;
//            }
//            out << endl;
//        }
//    }
//
//
//}


} //end MonteRay namespace

