#ifndef MONTERAY_READLNK3DNT_HH_
#define MONTERAY_READLNK3DNT_HH_

#include <iosfwd>
#include <string>
#include <vector>
#include <set>
#include <cassert>

#include "MonteRay_BinaryReadFcns.hh"


namespace MonteRay {

/// \brief  Reads LNK3DNT binary files.

/// \details MCATK reader for the LNK3DNT binary files. The LNK3DNT file is a binary,\n
/// code dependent file to enable the mixing of marcroscopic cross sections on the\n
/// fine mesh by a volume fraction method.  It does not, however, contain the ZAIDs\n
/// present in each material or the boundary conditions for the mesh.  This info \n
/// must be brought in by another input deck or hardwired.\n\n
/// <b>References:</b>\n
/// Alcouffe, R.E., Baker, R.S., Dahl, J.A., Turner, S.A., and Ward, R.C., (2005),\n
/// "PARTISN: A Time-Dependent, Parallel Neutral Particle Transport Code System (A Manual v5.34)",\n
/// LANL Report LA-UR-05-3925, p11-69.\n

class MonteRay_ReadLnk3dnt
{
public:
    enum GEOM  { SPHERICAL=3, RZ=7, XYZ=14, RZT=15 };

    typedef int    Partisn_int;
    typedef double Partisn_real;

    MonteRay_ReadLnk3dnt( const std::string& );
   ~MonteRay_ReadLnk3dnt( void ) {}

    void ReadMatData();


    void Print( std::ostream& ) const;

	Partisn_int getFileVersion ( void ) const  { return FileVersion;  }
    Partisn_int getGeometrySpec( void ) const  { return GeometrySpec; }
    std::string getGeometryString( void ) const;
    Partisn_int getBlockLevels ( void ) const  { return BlockLevels;  }
    size_t getNUniqueMatIDs( void ) const { return UniqueMatIDs.size(); }
    Partisn_int getMaxMatID( void ) const;
    size_t getDim(void) const { return Dim; }
    std::set< Partisn_int > getMaterialList( void ) const { return UniqueMatIDs; }

    /// NCells: Data Object function that returns the number of fine mesh cells in a given dimension (XYZ)
    /// \param d Dimension index
	unsigned NCells( unsigned d )        const { return nCells[ d ]; }
    double   minVertex( unsigned d ) const { return VertexPositions[ d ].front(); }
    double   maxVertex( unsigned d ) const { return VertexPositions[ d ].back(); }

    /// MaxMaterialsPerCell: Data Object function that returns the maximum number of materials used in any mesh cell.
	///   Each mesh cell may contain a different number and listing of materials.
	unsigned MaxMaterialsPerCell( void ) const { return MaxNMatsInCell; }

	template<typename T>
	void fillVertexArray  ( unsigned, T* ) const;

	template<typename T>
	void fillDensityArray ( T* )           const;

	template<typename T>
	void fillMaterialArray( T* )              const;

	std::vector< Partisn_real > getVertices( unsigned index ) {
		assert(index<Dim);
		return VertexPositions[index];
	}

    ///  The function extract will fill in two arrays, the MatID and the density array based on the View object passed in.\n
//    template <unsigned DIM>
//    void extract( std::vector<int> &matid,
//                  std::vector<double> &density,
//                  const DomainView<DIM> &view ) {
//    	std::runtime_error( "MonteRay_ReadLnk3dnt::extract -- function disabled in MonteRay.");
//
//        readBlock( "beginMatInfo", view, matid );
//
//        /* Advance file to densities */
//        unsigned NBytesForMatID = view.domainSize() * MaxNMatsInCell * sizeof( Partisn_int );
//        file.useMark( "beginMatInfo" );
//        file.advFile(  NBytesForMatID );
//        file.setMark( "beginDensity");
//
//        readBlock( "beginDensity", view, density );
//    }

    unsigned calcNumTotalCells(void ) const {
        unsigned total = 1;
        for( unsigned i=0; i < MAXDIM; ++i ) {
            total *= nCells[i];
        }
        return total;
    }

    unsigned getNumTotalCells(void) const { return TotalNumCells; }

    Partisn_real getDensity( unsigned cell, unsigned matNum ) const {
        return Density[cell + matNum*TotalNumCells];
    }

    Partisn_int getMatID( unsigned cell, unsigned matNum ) const {
        return MatIDs[cell + matNum*TotalNumCells];
    }

private:

     enum DIMEN { OneD=1, TwoD=2, ThreeD=3   } Dim;

     /// <i>CharLength = 8</i> is implicitly REQUIRED by PARTISN for LNK3DNT
     static const unsigned CharLength = 8;

     /// <i>MAXDIM = 3</i> is the maximum dimensions.
     static const unsigned MAXDIM = 3;

     /// <i>FileVersion</i> is IVERS for LNK3DNT documentation. MCATK uses 5.34, earlier versions may not be compatible.
     Partisn_int FileVersion;
     /// <i>Geometry</i> is IGOM in LNK3DNT documentation. Options are XY (6 in lnk3dnt file) , XYZ (14)
     Partisn_int GeometrySpec;
     /// <i>NZones</i> is NZONE in LNK3DNT documentation. Each zone is homogeneous for neutron.  Ie. no interface treatments.
     Partisn_int NZones;
     /// <i>nCells</i> is NINTI, NINTJ, NINTK in LNK3DNT documentation.
     Partisn_int nCells[ MAXDIM ];
     /// <i>MaxNMatsInCell</i> is NMXSP in LNK3DNT documentation.
     Partisn_int MaxNMatsInCell;
     /// <i>BlockLevels</i> is ILEVEL in LNK3DNT documentation.
     Partisn_int BlockLevels;

     std::vector< Partisn_int  > MatIDs;
     std::vector< Partisn_real > Density;
     std::vector< Partisn_real > VertexPositions[ MAXDIM ];

     std::set< Partisn_int > UniqueMatIDs;
     MonteRay_BinaryReader file;

     unsigned TotalNumCells;

     void ReadGeomData();

//     template<unsigned DIM, typename T >
//     unsigned readBlock( const std::string& blockMark, const DomainView<DIM>& view, std::vector<T>& data ) {
//         std::vector<T> temp( view.size() * MaxNMatsInCell );
//         unsigned TotalCells = view.domainSize();
//         typename std::vector<T>::iterator ptr = temp.begin();
//         for(unsigned j=0; j< MaxNMatsInCell; ++j) {
//             file.useMark( blockMark );
//             file.advFile( j * TotalCells * sizeof(T) );
//
//             unsigned int k = *view.begin();
//             file.advFile( *view.begin() * sizeof(T) );
//             for(typename DomainView<DIM>::const_iterator i=view.begin(); i != view.end(); ++i, ++ptr) {
//                 unsigned int skip=*i - k;
//                 if( skip > 1 ) {
//                     file.advFile( (skip-1) * sizeof(T) );
//                 }
//                 file.read( *ptr );
//                 k=*i;
//             }
//         }
//         data.swap( temp );
//
//         unsigned BytesRead = TotalCells * sizeof( T );
//         return BytesRead;
//     }
};

/// fillVertexArray: Data Object function that can be used by the mesh object to fill the vertex array

/// \param d Dimension index
/// \param vertex A pointer to an array that will hold one dimension of the vertex position.
template<typename T>
void
MonteRay_ReadLnk3dnt::fillVertexArray( unsigned d, T* vertex ) const {
    for( size_t i=0; i<VertexPositions[d].size(); ++i ) {
        vertex[ i ] = VertexPositions[d][ i ];
    }
}

/// fillDensityArray: Data Object function that can be used by the mesh object to fill the density array

/// \param density - A pointer to an array that will hold the densities. The array will be filled the fastest in this order, by x-axis(i), y-axis(j), z-axis(k), mass(m).
/// \param
template<typename T>
void
MonteRay_ReadLnk3dnt::fillDensityArray( T *density ) const {
    // Check that geometry info has been read
    // Check if density info is read, if not scream
    if (Density.empty()) {  // Change in case this is an empty domain
        std::string msg;
        msg = "Density array is empty! \nFound while processing the lnk3dnt file.  Has the function MonteRay_ReadLnk3dnt::ReadMatData been called?\n";
        throw std::runtime_error( msg.c_str() );
    }
    for( size_t i=0; i<Density.size(); ++i ) {
        density[i] = Density[i];
    }
}

/// fillMateralArray: Data Object function can be used by the mesh object to fill the material array

/// \param materialIDs - A pointer to an array that will hold the material IDs. The array will be filled the fastest in this order, by x-axis(i), y-axis(j), z-axis(k), mass(m).
/// \param
template<typename T>
void
MonteRay_ReadLnk3dnt::fillMaterialArray( T *materialIDs ) const {
    // Check that geometery info has been read
    // Check if matid info is read, if not scream
    if (MatIDs.empty()) {  // Change in case this is an empty domain
        std::string msg;
        msg = "Material ID array is empty! \nFound while processing the lnk3dnt file.  Has the function MonteRay_ReadLnk3dnt::ReadMatData been called?\n";
        throw std::runtime_error( msg.c_str() );
    }

    for( size_t i=0; i<MatIDs.size(); ++i ) {
        materialIDs[i] = MatIDs[i];
    }
}

} // end namespace

#endif /*READLNK3DNT_HH_*/
