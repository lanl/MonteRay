/*
 * MonteRay_GridBins.cc
 *
 *  Created on: Jan 30, 2018
 *      Author: jsweezy
 */

#include "MonteRay_GridBins.hh"
#include "MonteRay_binaryIO.hh"

#ifndef __CUDA_ARCH__
#include <stdexcept>
#include <sstream>
#endif

#include <fstream>

#include "BinarySearch.hh"

namespace MonteRay {


void
MonteRay_GridBins::initialize( const std::vector<gpuFloatType_t>& bins ) {
#ifndef __CUDA_ARCH__
	verticesVec = new std::vector<gpuFloatType_t>;
	*verticesVec = bins;
    setup();
#endif
}

void
MonteRay_GridBins::initialize( gpuFloatType_t min, gpuFloatType_t max, unsigned nBins){
#ifndef __CUDA_ARCH__
	if( debug ) printf( "Debug: MonteRay_GridBins::initialize -- min =%f\n", min);
	if( debug ) printf( "Debug: MonteRay_GridBins::initialize -- max =%f\n", max);
	if( debug ) printf( "Debug: MonteRay_GridBins::initialize -- nBins =%d\n", nBins);
	verticesVec = new std::vector<gpuFloatType_t>();
	if( debug ) printf( "Debug: MonteRay_GridBins::initialize -- verticesVec =%d\n", verticesVec);
	delta = (max-min)/nBins;
	if( debug ) printf( "Debug: MonteRay_GridBins::initialize -- delta=%f\n", delta);
    gpuFloatType_t vertex = min;
    for( unsigned i = 0; i<nBins+1; ++i ) {
    	if( debug ) printf( "Debug: MonteRay_GridBins::initialize -- vertex[%d]=%f\n",i, vertex);
    	verticesVec->push_back( vertex );
        vertex += delta;
    }
    if( debug ) printf( "Debug: MonteRay_GridBins::initialize -- calling setup()\n");
    setup();
#endif
}

void
MonteRay_GridBins::removeVertex(unsigned i) {
#ifndef __CUDA_ARCH__
	verticesVec->erase( verticesVec->begin() + i );
    setup();
#endif
}

void
MonteRay_GridBins::setup(void) {
#ifndef __CUDA_ARCH__
	if( debug ) printf( "Debug: MonteRay_GridBins::setup -- verticesVec->size() = %d\n", verticesVec->size());
	if( verticesVec->size() == 1 ) {
		minVertex = 0.0;
		maxVertex = verticesVec->front();
		numBins = 1;
	} else {
		minVertex = verticesVec->front();
		maxVertex = verticesVec->back();
		numBins = verticesVec->size() - 1;
	}
	// for now limit the number of vertices due to fixed memory requirements
	MONTERAY_VERIFY( verticesVec->size() <= MAXNUMVERTICES, "MonteRay_GridBins::setup -- number of vertices exceeds the max size: MAXNUMVERTICES" )

    validate();
#endif
}

void
MonteRay_GridBins::modifyForRadial(void) {
#ifndef __CUDA_ARCH__
    if( radialModified ) return;
    radialModified = true;
    type = RADIAL;

    // test for negative
    for( unsigned i=0; i< verticesVec->size(); ++i) {
        if( verticesVec->at(i) < 0.0 ) {
            std::stringstream msg;
            msg << " Radial bin edge values must be non-negative!!! " << std::endl
                << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_GridBins::modifyForRadial" << std::endl << std::endl;

            throw std::runtime_error( msg.str() );
        }
    }

    // Remove any zero entry
    if( verticesVec->at(0) == 0.0 ) {
        removeVertex(0);
    }

    // store the vertices values squared
    verticesSqVec = new std::vector<gpuFloatType_t>;
    for( unsigned i=0; i< verticesVec->size(); ++i) {
        gpuFloatType_t value = verticesVec->at(i);
        verticesSqVec->push_back( value*value );
    }
    numBins = verticesVec->size();
    validate();
#endif
}

void
MonteRay_GridBins::validate() {
#ifndef __CUDA_ARCH__
    if( minVertex >= maxVertex ) {
        std::stringstream msg;
        msg << " The minimum vertex must be less than the maximum vertex !!! " << std::endl
            << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_GridBins::modifyForRadial"  << std::endl << std::endl;

        throw std::runtime_error( msg.str() );
    }
    if( numBins == 0 ) {
        std::stringstream msg;
        msg << " The number of bins must be greater than 0 !!! " << std::endl
            << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_GridBins::modifyForRadial"  << std::endl << std::endl;

        throw std::runtime_error( msg.str() );
    }

    // test ascending
    for( unsigned i=1; i< verticesVec->size(); ++i) {
        if( verticesVec->at(i) <= verticesVec->at(i-1) ) {
            std::stringstream msg;
            msg << " The bin edge values must be ascending!!! " << std::endl
                << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRay_GridBins::modifyForRadial"  << std::endl << std::endl;

            throw std::runtime_error( msg.str() );
        }
    }

    if( verticesVec ) {
    	vertices = const_cast<gpuFloatType_t*>( verticesVec->data() );
    	nVertices = verticesVec->size();
    }

    if( verticesSqVec ) {
    	verticesSq = const_cast<gpuFloatType_t*>( verticesSqVec->data() );
    	nVerticesSq = verticesSqVec->size();
    }
#endif
}

void MonteRay_GridBins::write( const std::string& filename ) {
    std::ofstream outfile;

    outfile.open( filename.c_str(), std::ios::binary | std::ios::out);
    if( ! outfile.is_open() ) {
        fprintf(stderr, "MonteRay_GridBins::write -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( outfile.good() );
    outfile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    write( outfile );
    outfile.close();
}

void MonteRay_GridBins::read( const std::string& filename ) {
    std::ifstream infile;
    if( infile.is_open() ) {
        infile.close();
    }
    infile.open( filename.c_str(), std::ios::binary | std::ios::in);

    if( ! infile.is_open() ) {
        fprintf(stderr, "Debug:  MonteRay_GridBins::read -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
        exit(1);
    }
    assert( infile.good() );
    infile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
    read(infile);
    infile.close();
}

void
MonteRay_GridBins::write(std::ostream& outf) const {
	unsigned version = 0;

	binaryIO::write(outf, version );
	binaryIO::write(outf, minVertex);
	binaryIO::write(outf, maxVertex);
	binaryIO::write(outf, numBins );
	binaryIO::write(outf, delta );

	binaryIO::write(outf, nVertices );
	for( unsigned i=0; i< nVertices; ++i ){
		binaryIO::write(outf, vertices[i] );
	}

	binaryIO::write(outf, nVerticesSq );
	for( unsigned i=0; i< nVerticesSq; ++i ){
		binaryIO::write(outf, verticesSq[i] );
	}

	binaryIO::write(outf, type );
	binaryIO::write(outf, radialModified );
}

void
MonteRay_GridBins::read(std::istream& infile) {
	unsigned version;
	binaryIO::read(infile, version );

	if( version == 0 ) {
		read_v0(infile);
	}
	validate();
}

void
MonteRay_GridBins::read_v0(std::istream& infile){
	binaryIO::read(infile, minVertex);
	binaryIO::read(infile, maxVertex);
	binaryIO::read(infile, numBins );
	binaryIO::read(infile, delta );

	binaryIO::read(infile, nVertices );
	if( verticesVec ) {
		verticesVec->clear();
		delete verticesVec;
	}
	if( nVertices > 0 ) {
		verticesVec = new std::vector<gpuFloatType_t>;
	}
	for( unsigned i=0; i< nVertices; ++i ){
		gpuFloatType_t vertex;
		binaryIO::read(infile, vertex );
		verticesVec->push_back( vertex );
	}

	binaryIO::read(infile, nVerticesSq );
	if( verticesSqVec ) {
		verticesSqVec->clear();
		delete verticesSqVec;
	}

	if( nVerticesSq > 0 ) {
		verticesSqVec = new std::vector<gpuFloatType_t>;
	}
	for( unsigned i=0; i< nVerticesSq; ++i ){
		gpuFloatType_t vertexSq;
		binaryIO::read(infile, vertexSq );
		verticesSqVec->push_back( vertexSq );
	}

	binaryIO::read(infile, type );
	binaryIO::read(infile, radialModified );
}

CUDA_CALLABLE_MEMBER int
MonteRay_GridBins::getLinearIndex(gpuFloatType_t pos) const {
    // returns -1 for one neg side of mesh
    // and number of bins on the pos side of the mesh
    // need to call isIndexOutside(dim, grid, index) to check if the
    // index is in the mesh

    int dim_index;
    if( pos <= minVertex ) {
        dim_index = -1;
    } else if( pos >= maxVertex ) {
        dim_index = getNumBins();
    } else {
       	dim_index = LowerBoundIndex( vertices, nVertices, pos);
    }
    return dim_index;
}

CUDA_CALLABLE_MEMBER int
MonteRay_GridBins::getRadialIndexFromRSq( gpuFloatType_t rSq) const {
	MONTERAY_ASSERT( rSq >= 0.0 );
	MONTERAY_ASSERT( radialModified );

    gpuFloatType_t max = verticesSq[nVerticesSq-1];

    int radialIndex;
    if( rSq >= max ) {
        radialIndex = getNumBins();
    } else {
    	radialIndex = UpperBoundIndex( verticesSq, nVerticesSq, rSq);
    }
    return radialIndex;
}


} /* namespace MonteRay */
