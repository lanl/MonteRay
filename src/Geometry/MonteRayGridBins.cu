/*
 * MonteRayGridBins.cc
 *
 *  Created on: Jan 30, 2018
 *      Author: jsweezy
 */

#include "MonteRayGridBins.hh"
#include "MonteRay_binaryIO.hh"

#ifndef __CUDA_ARCH__
#include <stdexcept>
#include <sstream>
#endif

#include "BinarySearch.hh"

namespace MonteRay {


void
MonteRayGridBins::initialize( const std::vector<gpuFloatType_t>& bins ) {
#ifndef __CUDA_ARCH__
	verticesVec = new std::vector<gpuFloatType_t>;
	*verticesVec = bins;
    setup();
#endif
}

void
MonteRayGridBins::initialize( gpuFloatType_t min, gpuFloatType_t max, unsigned nBins){
#ifndef __CUDA_ARCH__
	verticesVec = new std::vector<gpuFloatType_t>;
	delta = (max-min)/nBins;
    double vertex = min;
    for( unsigned i = 0; i<nBins+1; ++i ) {
    	verticesVec->push_back( vertex );
        vertex += delta;
    }
    setup();
#endif
}

void
MonteRayGridBins::removeVertex(unsigned i) {
#ifndef __CUDA_ARCH__
	verticesVec->erase( verticesVec->begin() + i );
    setup();
#endif
}

void
MonteRayGridBins::setup(void) {
#ifndef __CUDA_ARCH__
	if( verticesVec->size() == 1 ) {
		minVertex = 0.0;
		maxVertex = verticesVec->front();
		numBins = 1;
	} else {
		minVertex = verticesVec->front();
		maxVertex = verticesVec->back();
		numBins = verticesVec->size() - 1;
	}

    validate();
#endif
}

void
MonteRayGridBins::modifyForRadial(void) {
#ifndef __CUDA_ARCH__
    if( radialModified ) return;
    radialModified = true;
    type = RADIAL;

    // test for negative
    for( unsigned i=0; i< verticesVec->size(); ++i) {
        if( verticesVec->at(i) < 0.0 ) {
            std::stringstream msg;
            msg << " Radial bin edge values must be non-negative!!! " << std::endl
                << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRayGridBins::modifyForRadial" << std::endl << std::endl;

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
        double value = verticesVec->at(i);
        verticesSqVec->push_back( value*value );
    }
    numBins = verticesVec->size();
    validate();
#endif
}

void
MonteRayGridBins::validate() {
#ifndef __CUDA_ARCH__
    if( minVertex >= maxVertex ) {
        std::stringstream msg;
        msg << " The minimum vertex must be less than the maximum vertex !!! " << std::endl
            << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRayGridBins::modifyForRadial"  << std::endl << std::endl;

        throw std::runtime_error( msg.str() );
    }
    if( numBins == 0 ) {
        std::stringstream msg;
        msg << " The number of bins must be greater than 0 !!! " << std::endl
            << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRayGridBins::modifyForRadial"  << std::endl << std::endl;

        throw std::runtime_error( msg.str() );
    }

    // test ascending
    for( unsigned i=1; i< verticesVec->size(); ++i) {
        if( verticesVec->at(i) <= verticesVec->at(i-1) ) {
            std::stringstream msg;
            msg << " The bin edge values must be ascending!!! " << std::endl
                << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "MonteRayGridBins::modifyForRadial"  << std::endl << std::endl;

            throw std::runtime_error( msg.str() );
        }
    }

    if( verticesVec ) {
    	vertices = const_cast<gpuFloatType_t*>( verticesVec->data() );
    	nVertices = verticesVec->size();
    }

    if( verticesSqVec ) {
    	verticesSq = const_cast<gpuFloatType_t*>( verticesSqVec->data() );
    	nVerticesSq = verticesVec->size();
    }
#endif
}

void
MonteRayGridBins::write(std::ostream& outf) const {
	unsigned version = 0;
	binaryIO::write(outf, version );

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
MonteRayGridBins::read(std::istream& infile) {
	unsigned version;
	binaryIO::read(infile, version );

	if( version == 0 ) {
		read_v0(infile);
	}
}

void
MonteRayGridBins::read_v0(std::istream& infile){
	binaryIO::read(infile, minVertex);
	binaryIO::read(infile, maxVertex);
	binaryIO::read(infile, numBins );
	binaryIO::read(infile, delta );

	binaryIO::read(infile, nVertices );
	for( unsigned i=0; i< nVertices; ++i ){
		binaryIO::read(infile, vertices[i] );
	}

	binaryIO::read(infile, nVerticesSq );
	for( unsigned i=0; i< nVerticesSq; ++i ){
		binaryIO::read(infile, verticesSq[i] );
	}

	binaryIO::read(infile, type );
	binaryIO::read(infile, radialModified );
}

CUDA_CALLABLE_MEMBER int
MonteRayGridBins::getLinearIndex(gpuFloatType_t pos) const {
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
MonteRayGridBins::getRadialIndexFromRSq( gpuFloatType_t rSq) const {
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
