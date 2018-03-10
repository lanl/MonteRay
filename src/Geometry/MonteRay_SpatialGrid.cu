/*
 * MonteRay_SpatialGrid.cc
 *
 *  Created on: Feb 5, 2018
 *      Author: jsweezy
 */

#include "MonteRay_SpatialGrid.hh"
#include "MonteRay_CartesianGrid.hh"
#include "MonteRay_binaryIO.hh"

#ifndef __CUDA_ARCH__
#include <stdexcept>
#include <sstream>
#endif

#include <fstream>


namespace MonteRay {

const unsigned MonteRay_SpatialGrid::OUTSIDE_MESH = UINT_MAX;

CUDAHOST_CALLABLE_MEMBER
MonteRay_SpatialGrid::MonteRay_SpatialGrid() :
	CopyMemoryBase<MonteRay_SpatialGrid>(),
    CoordinateSystem(TransportMeshTypeEnum::NONE),
    dimension( 0 ),
    initialized( false )
{
		pGridInfo[0] = nullptr;
		pGridInfo[1] = nullptr;
		pGridInfo[2] = nullptr;
		pGridSystem = nullptr;
}

CUDAHOST_CALLABLE_MEMBER
MonteRay_SpatialGrid::MonteRay_SpatialGrid( const MonteRay_SpatialGrid& rhs ) :
		CopyMemoryBase<MonteRay_SpatialGrid>(),
        CoordinateSystem( rhs.CoordinateSystem ),
        dimension( rhs.dimension ),
        initialized( false )
//        transform( rhs.transform )
{
	pGridInfo[0] = rhs.pGridInfo[0];
	pGridInfo[1] = rhs.pGridInfo[1];
	pGridInfo[2] = rhs.pGridInfo[2];
	pGridSystem = rhs.pGridSystem;
    if( rhs.initialized ) {
        initialize();
    }
}

//  Disable assignment operator until needed
//MonteRay_SpatialGrid&
//MonteRay_SpatialGrid::operator=( const MonteRay_SpatialGrid& rhs ){
//    if( initialized ){
//#ifndef __CUDA_ARCH__
//        throw std::runtime_error("SpatialGrid::operator= -- Cannot use the assignment operator to assign to a SpatialGrid that has already been initialized.");
//#else
//        ABORT( "SpatialGrid::operator= -- Cannot use the assignment operator to assign to a SpatialGrid that has already been initialized.\n" );
//#endif
//    }
//
//    CoordinateSystem = rhs.CoordinateSystem;
//    dimension = rhs.dimension;
//    for( auto i=0; i<dimension; ++i) {
//    	if( pGridInfo[i] ) delete pGridInfo[i];
//    	pGridInfo[i] = new GridBins_t();
//    	pGridInfo[i] = rhs.pGridInfo[i];
//    }
//
//   // transform = rhs.transform;
//
//    if( rhs.initialized ) {
//        initialize();
//    }
//
//    return *this;
//}

CUDAHOST_CALLABLE_MEMBER
void
MonteRay_SpatialGrid::setCoordinateSystem(TransportMeshTypeEnum::TransportMeshTypeEnum_t system) {
    if( !( system < TransportMeshTypeEnum::MAX ) ) {
#ifndef __CUDA_ARCH__
    	std::stringstream msg;
        msg << " Equal or Greater than TransportMeshTypeEnum::MAX !!! " << std::endl
            << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_SpatialGrid::setCoordinateSystem" << std::endl << std::endl;

//        throw SpatialGridException( SpatialGridException::INITIALIZATION_ERROR, msg);
        throw std::runtime_error( msg.str() );
#else
        ABORT( "Equal or Greater than TransportMeshTypeEnum::MAX !!!\n" );
#endif
    }

    CoordinateSystem=system;
}

CUDAHOST_CALLABLE_MEMBER
void MonteRay_SpatialGrid::initialize(void) {
    if( initialized ) {
#ifndef __CUDA_ARCH__
        std::stringstream msg;
        msg << " Class already initialized, can't re-initialize !!! " << std::endl
            << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_SpatialGrid::initialize" << std::endl << std::endl;

        //throw SpatialGridException( SpatialGridException::INITIALIZATION_ERROR, msg);
        throw std::runtime_error( msg.str() );
#else
        ABORT( "MonteRay_SpatialGrid:: initialize -- Class instance already initialized, can't re-initialize !!! \n" );
#endif
    }
    initialized = true;

    if( dimension == 0 ) {
#ifndef __CUDA_ARCH__
        std::stringstream msg;
        msg << " Number of dimensions is not set!!! " << std::endl
            << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_SpatialGrid::initialize" << std::endl << std::endl;

        //throw SpatialGridException( SpatialGridException::INITIALIZATION_ERROR, msg);
        throw std::runtime_error( msg.str() );
#else
        ABORT( "MonteRay_SpatialGrid:: initialize -- Number of dimensions is not set!!!\n" );
#endif
    }

    if( CoordinateSystem == TransportMeshTypeEnum::NONE ) {
#ifndef __CUDA_ARCH__
        std::stringstream msg;
        msg << " Coordinate system is not set!!! " << std::endl
            << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_SpatialGrid::initialize" << std::endl << std::endl;

        //throw SpatialGridException( SpatialGridException::INITIALIZATION_ERROR, msg);
        throw std::runtime_error( msg.str() );
#else
        ABORT( "MonteRay_SpatialGrid:: initialize -- Coordinate system is not set!!!\n" );
#endif
    }

    // check grid info is allocated
    for( unsigned d = 0; d < dimension; ++d ){
        if( ! pGridInfo[d] ) {
#ifndef __CUDA_ARCH__
        	std::stringstream msg;
            msg << "Grid data for index [" << std::to_string(d).c_str()  << "] is not allocated! " << std::endl
                << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_SpatialGrid::initialize" << std::endl << std::endl;

            //throw SpatialGridException( SpatialGridException::INITIALIZATION_ERROR, msg );
            throw std::runtime_error( msg.str() );
#else
            ABORT( "MonteRay_SpatialGrid:: initialize -- Grid data is not allocated!!!\n" );
#endif
            break;
        }
    }

    for( unsigned d = 0; d < dimension; ++d ){
        if( getNumGridBins(d) == 0 ) {
#ifndef __CUDA_ARCH__
        	std::stringstream msg;
            msg << " Grid vertices for index [" << std::to_string(d).c_str()  << "] is not set! " << std::endl
                << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_SpatialGrid::initialize" << std::endl << std::endl;

            //throw SpatialGridException( SpatialGridException::INITIALIZATION_ERROR, msg );
            throw std::runtime_error( msg.str() );
#else
            ABORT( "MonteRay_SpatialGrid:: initialize -- Grid vertices is not set!!!\n" );
#endif
            break;
        }
    }

    //TODO: TRA - need 1d,2d for Cart_regular and Cart?
    switch (CoordinateSystem) {
//        case TransportMeshTypeEnum::Cartesian_Regular:
//            pGridSystem.reset( new CartesianGrid(3,gridInfo) );
//            dynamic_cast<CartesianGrid*>( pGridSystem.get() )->setRegular();
//            break;

        case TransportMeshTypeEnum::Cartesian:
        	if( pGridSystem ) delete pGridSystem;
        	pGridSystem = new MonteRay_CartesianGrid(3,pGridInfo);
            break;

//        case TransportMeshTypeEnum::Cylindrical:
//            if(dimension < MaxDim ) {
//                pGridSystem.reset( new MonteRay_CylindricalGrid(dimension,gridInfo) );
//                break;
//            } else {
//#ifndef __CUDA_ARCH__
//                std::stringstream msg;
//                msg << " Not Permitted Yet. Dimension set to 3 with Cylindrical. " << std::endl
//                    << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << BOOST_CURRENT_FUNCTION << std::endl << std::endl;
//
//                throw SpatialGridException( SpatialGridException::DIMENSION_ERROR, msg );
//#else
//                ABORT( "MonteRay_SpatialGrid:: initialize -- Not Permitted Yet. Dimension set to 3 with Cylindrical.!!!\n" );
//#endif
//                break;
//            }

//        case TransportMeshTypeEnum::Spherical:
//            pGridSystem.reset( new MonteRay_SphericalGrid(1,gridInfo) );
//            break;

        default:
#ifndef __CUDA_ARCH__
            std::stringstream msg;
            msg << " Unknown coordinate system or coordinate system is not set! " << std::endl
                << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_SpatialGrid::initialize" << std::endl << std::endl;

            //throw SpatialGridException( SpatialGridException::INITIALIZATION_ERROR, msg );
            throw std::runtime_error( msg.str() );
#else
            ABORT( "MonteRay_SpatialGrid:: initialize -- Unknown coordinate system or coordinate system is not set!!!\n" );
#endif
            break;
    }

    // Test for too many cells
    getNumCells();
}

CUDAHOST_CALLABLE_MEMBER
void
MonteRay_SpatialGrid::setDimension( unsigned dim) {
    checkDim( dim );
    dimension = dim;
}

CUDAHOST_CALLABLE_MEMBER
void
MonteRay_SpatialGrid::setGrid( unsigned index, gpuFloatType_t min, gpuFloatType_t max, unsigned numBins ) {
    checkDim( index+1 );
    if( debug ) printf( "Debug: MonteRay_SpatialGrid::setGrid -- index =%d\n", index);
    if( pGridInfo[index] ) delete pGridInfo[index];
    pGridInfo[index] = new GridBins_t();
    pGridInfo[index]->initialize( min, max, numBins);
}

CUDAHOST_CALLABLE_MEMBER
void
MonteRay_SpatialGrid::setGrid( unsigned index, const std::vector<gpuFloatType_t>& vertices ) {
    checkDim( index+1 );
    if( pGridInfo[index] ) delete pGridInfo[index];
    pGridInfo[index] = new GridBins_t();
    pGridInfo[index]->initialize( vertices );
}

CUDA_CALLABLE_MEMBER
unsigned
MonteRay_SpatialGrid::getNumGridBins( unsigned index ) const {
    checkDim( index+1 );
#ifdef __CUDA_ARCH__
    //if( debug ) printf("Debug: MonteRay_SpatialGrid::getNumGridBins -- Device pGridInfo[%d]=%p\n",index, pGridInfo[index]);
#else
    //if( debug ) printf("Debug: MonteRay_SpatialGrid::getNumGridBins -- Host pGridInfo[%d]=%p\n",index, pGridInfo[index]);
#endif

    return pGridInfo[index]->getNumBins();
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t
MonteRay_SpatialGrid::getMinVertex( unsigned index ) const {
    checkDim( index+1 );
    return pGridInfo[index]->getMinVertex();
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t
MonteRay_SpatialGrid::getMaxVertex( unsigned index ) const {
    checkDim( index+1 );
    return pGridInfo[index]->getMaxVertex();
}

CUDA_CALLABLE_MEMBER
gpuFloatType_t
MonteRay_SpatialGrid::getDelta(unsigned index) const {
    checkDim( index+1 );
    return pGridInfo[index]->getDelta();
}

CUDA_CALLABLE_MEMBER
void
MonteRay_SpatialGrid::checkDim( unsigned dim ) const {
    if( dim == 0 ) {
#ifndef __CUDA_ARCH__
        std::stringstream msg;
        msg << " Number of dimensions can not be 0 !!! " << std::endl
            << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_SpatialGrid::checkDim" << std::endl << std::endl;

        //throw SpatialGridException( SpatialGridException::INITIALIZATION_ERROR,  msg);
        throw std::runtime_error( msg.str() );
#else
        ABORT( "MonteRay_SpatialGrid::checkDim -- Number of dimensions can not be 0 !!!\n" );
#endif
    }

    if( dim > MaxDim ) {
#ifndef __CUDA_ARCH__
         std::stringstream msg;
         msg << " Dimension greater than MaxDim = 3 !!! " << std::endl
             << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_SpatialGrid::checkDim" << std::endl << std::endl;

         //throw SpatialGridException( SpatialGridException::INVALID_DIM_INDEX, msg);
         throw std::runtime_error( msg.str() );
#else
         ABORT( "MonteRay_SpatialGrid::checkDim -- Dimension greater than MaxDim = 3 !!!\n" );
#endif
     }

}

CUDA_CALLABLE_MEMBER
unsigned
MonteRay_SpatialGrid::getNumCells(void) const {
    if( dimension == 0 ) {
#ifndef __CUDA_ARCH__
        std::stringstream msg;
        msg << " Number of dimensions can not be 0 !!! " << std::endl
                << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_SpatialGrid::getNumCells" << std::endl << std::endl;

        //throw SpatialGridException( SpatialGridException::INITIALIZATION_ERROR,  msg);
        throw std::runtime_error( msg.str() );
#else
        ABORT( "MonteRay_SpatialGrid::getNumCells -- Number of dimensions can not be 0 !!!\n" );
#endif
    }

    unsigned long long int nCells = 1;
    for( auto d=0; d < dimension; ++d  ){
        nCells *= getNumGridBins(d);
    }
    if( nCells > UINT_MAX ) {
#ifndef __CUDA_ARCH__
        std::stringstream msg;
        msg << " Number of cells exceeds the capacity of an unsigned int !!! " << std::endl
                << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_SpatialGrid::getNumCells" << std::endl << std::endl;

        //throw SpatialGridException( SpatialGridException::INITIALIZATION_ERROR,  msg);
        throw std::runtime_error( msg.str() );
#else
        ABORT( "MonteRay_SpatialGrid::getNumCells -- Number of cells exceeds the capacity of an unsigned int !!!\n" );
#endif
    }
    return unsigned(nCells);
}

//void
//MonteRay_SpatialGrid::setTransformation( const mcatk::Transformation& T){
//    if( transform ) {
//        std::stringstream msg;
//               msg << " Transformation already set !!! " << std::endl
//                       << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << BOOST_CURRENT_FUNCTION << std::endl << std::endl;
//
//               throw SpatialGridException( SpatialGridException::INITIALIZATION_ERROR,  msg);
//    } else {
//        transform = T;
//    }
//}
//
//const mcatk::Transformation*
//MonteRay_SpatialGrid::getTransformation( void ) const {
//    if( transform ) { return &(*transform); }
//    return nullptr;
//}

CUDA_CALLABLE_MEMBER
gpuFloatType_t
MonteRay_SpatialGrid::getVertex(unsigned d, unsigned i ) const {
    if( d > dimension ) {
#ifndef __CUDA_ARCH__
         std::stringstream msg;
         msg << " Dimension index greater than number of set dimensions = " << dimension << "!" << std::endl
             << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_SpatialGrid::getVertex" << std::endl << std::endl;

         //throw SpatialGridException( SpatialGridException::INVALID_DIM_INDEX, msg);
         throw std::runtime_error( msg.str() );
#else
         ABORT( "MonteRay_SpatialGrid::getVertex -- Dimension index greater than number of set dimensions!!!\n" );
#endif
    }
    return pGridInfo[d]->vertices[i];
}

CUDA_CALLABLE_MEMBER
size_t MonteRay_SpatialGrid::getNumVertices(unsigned i) const {
	return pGridInfo[i]->getNumVertices();
}

CUDA_CALLABLE_MEMBER
size_t MonteRay_SpatialGrid::getNumVerticesSq(unsigned i) const{
	return pGridInfo[i]->getNumVerticesSq();
}

void MonteRay_SpatialGrid::write( const std::string& filename ) {
    std::ofstream outfile;

    outfile.open( filename.c_str(), std::ios::binary | std::ios::out);
    if( ! outfile.is_open() ) {
        fprintf(stderr, "MonteRay_SpatialGrid::write -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( outfile.good() );
    outfile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    write( outfile );
    outfile.close();
}

void MonteRay_SpatialGrid::read( const std::string& filename ) {
    std::ifstream infile;
    if( infile.is_open() ) {
        infile.close();
    }
    infile.open( filename.c_str(), std::ios::binary | std::ios::in);

    if( ! infile.is_open() ) {
        fprintf(stderr, "Debug:  MonteRay_SpatialGrid::read -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( infile.good() );
    infile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    read(infile);
    infile.close();
}

void
MonteRay_SpatialGrid::write(std::ostream& outf) const {
	unsigned version = 0;
	binaryIO::write(outf, version );

	binaryIO::write(outf, CoordinateSystem);
	binaryIO::write(outf, dimension);
	for( unsigned i = 0; i<dimension; ++i) {
		pGridInfo[i]->write(outf);
	}
}

void
MonteRay_SpatialGrid::read(std::istream& infile) {
	unsigned version;
	binaryIO::read(infile, version );

	if( version == 0 ) {
		read_v0(infile);
	}
	initialize();
}

void
MonteRay_SpatialGrid::read_v0(std::istream& infile){
	binaryIO::read(infile, CoordinateSystem);
	binaryIO::read(infile, dimension);
	for( unsigned i = 0; i<dimension; ++i) {
	    if( pGridInfo[i] ) delete pGridInfo[i];
	    pGridInfo[i] = new GridBins_t();
		pGridInfo[i]->read(infile);
	}
}

} /* namespace MonteRay */