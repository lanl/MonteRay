/*
 * MonteRay_GridBins.hh
 *
 *  Created on: Jan 30, 2018
 *      Author: jsweezy
 *
 *  A port of MCATK's GridBins class to MonteRay/CUDA
 */

#ifndef MonteRay_GridBins_HH_
#define MonteRay_GridBins_HH_

#ifndef __CUDA_ARCH__
#include <vector>
#endif

#include "MonteRayDefinitions.hh"
#include "MonteRayCopyMemory.hh"
#include "MonteRayVector3D.hh"
#include "GPUErrorCheck.hh"

#define MAXNUMVERTICES 1001

namespace MonteRay {

class MonteRay_GridBins : public CopyMemoryBase<MonteRay_GridBins>{
public:
	using Base = MonteRay::CopyMemoryBase<MonteRay_GridBins> ;
    //typedef MonteRay_GridBins[3] gridInfoArray_t;
    typedef MonteRay::Vector3D<gpuFloatType_t> Position_t;
    typedef MonteRay::Vector3D<gpuFloatType_t> Direction_t;

    enum coordinate_t{ LINEAR, RADIAL };

    CUDAHOST_CALLABLE_MEMBER MonteRay_GridBins(void) {
    	init();
    }


    CUDAHOST_CALLABLE_MEMBER MonteRay_GridBins( gpuFloatType_t min, gpuFloatType_t max, unsigned nBins) : CopyMemoryBase<MonteRay_GridBins>()  {
    	init();
        initialize( min, max, nBins );
    }

    CUDAHOST_CALLABLE_MEMBER MonteRay_GridBins( const std::vector<gpuFloatType_t>& bins ) {
    	init();
        initialize( bins );
    }

    MonteRay_GridBins&
    operator=( MonteRay_GridBins& rhs ) {
    	nVertices = rhs.nVertices;
    	nVerticesSq = rhs.nVerticesSq;
    	vertices = rhs.vertices;
    	verticesSq = rhs.verticesSq;
    	delta = rhs.delta;
    	minVertex = rhs.minVertex;
    	maxVertex = rhs.maxVertex;
    	numBins = rhs.numBins;
    	type = rhs.type;
    	radialModified = rhs.radialModified;

    	verticesVec = nullptr;
    	verticesSqVec = nullptr;
    	return *this;
    }

    //MonteRay_GridBins(const MonteRay_GridBins&);

    //MonteRay_GridBins& operator=( const MonteRay_GridBins& rhs );

    CUDAHOST_CALLABLE_MEMBER ~MonteRay_GridBins(void){

		if( Base::isCudaIntermediate ) {
			//std::cout << "Debug: MonteRay_GridBins::~MonteRay_GridBins -- isCudaIntermediate \n";
			MonteRayDeviceFree( vertices );
			MonteRayDeviceFree( verticesSq );
		} else {
			// if managed by std::vector on host - no free needed on host
			// TODO: use logic if vertices is allocated alone.
			// MonteRayHostFree( vertices, Base::isManagedMemory );
			// MonteRayHostFree( verticesSq, Base::isManagedMemory );
		}
		if( verticesVec ) {
			verticesVec->clear();
			delete verticesVec;
		}
		if( verticesSqVec ) {
			verticesSqVec->clear();
			delete verticesSqVec;
		}
    }

    CUDAHOST_CALLABLE_MEMBER std::string className(){ return std::string("MonteRay_GridBins");}

    CUDAHOST_CALLABLE_MEMBER void init() {
		nVertices = 0;
		nVerticesSq = 0;
		vertices = NULL;
		verticesSq = NULL;
		delta = 0.0;
		minVertex = 0.0;
		maxVertex = 0.0;
		numBins = 0;
		type = LINEAR;
		radialModified = false;

		verticesVec= NULL;
		verticesSqVec= NULL;
	}

    CUDAHOST_CALLABLE_MEMBER void copyToGPU(void) {
    	//if( debug ) std::cout << "Debug: MonteRay_GridBins::copyToGPU \n";
		Base::copyToGPU();
	}

    CUDAHOST_CALLABLE_MEMBER void copy(const MonteRay_GridBins* rhs) {
//		if( debug ) {
//			std::cout << "Debug: MonteRay_GridBins::copy(const MonteRay_GridBins* rhs) \n";
//		}

#ifdef __CUDACC__
		if( nVertices != 0 && (nVertices != rhs->nVertices) ) {
			std::cout << "Error: MonteRay_GridBins::copy -- can't change size of nVertices after initialization.\n";
			std::cout << "Error: MonteRay_GridBins::copy -- nVertices = " << nVertices << " \n";
			std::cout << "Error: MonteRay_GridBins::copy -- rhs->nVertices = " << rhs->nVertices << " \n";
			std::cout << "Error: MonteRay_GridBins::copy -- isCudaIntermediate = " << isCudaIntermediate << " \n";
			std::cout << "Error: MonteRay_GridBins::copy -- rhs->isCudaIntermediate = " << rhs->isCudaIntermediate << " \n";
			throw std::runtime_error("MonteRay_GridBins::copy -- can't change size after initialization.");
		}

		if( nVerticesSq != 0 && (nVerticesSq != rhs->nVerticesSq) ) {
			std::cout << "Error: MonteRay_GridBins::copy -- can't change size of nVerticesSq after initialization.\n";
			std::cout << "Error: MonteRay_GridBins::copy -- nVerticesSq = " << nVerticesSq << " \n";
			std::cout << "Error: MonteRay_GridBins::copy -- rhs->nVerticesSq = " << rhs->nVerticesSq << " \n";
			std::cout << "Error: MonteRay_GridBins::copy -- isCudaIntermediate = " << isCudaIntermediate << " \n";
			std::cout << "Error: MonteRay_GridBins::copy -- rhs->isCudaIntermediate = " << rhs->isCudaIntermediate << " \n";
			throw std::runtime_error("MonteRay_GridBins::copy -- can't change size after initialization.");
		}

		if( isCudaIntermediate ) {
			// host to device
			if( nVertices == 0 && rhs->nVertices > 0 ) {
				vertices = (gpuFloatType_t*) MONTERAYDEVICEALLOC( rhs->nVertices*sizeof(gpuFloatType_t), std::string("device - MonteRay_GridBins::vertices") );
			}
			if( nVerticesSq == 0 && rhs->nVerticesSq > 0) {
				verticesSq = (gpuFloatType_t*) MONTERAYDEVICEALLOC( rhs->nVerticesSq*sizeof(gpuFloatType_t), std::string("device - MonteRay_GridBins::verticesSq") );
			}
			MonteRayMemcpy( vertices,   rhs->vertices,   rhs->nVertices*sizeof(gpuFloatType_t), cudaMemcpyHostToDevice );
			MonteRayMemcpy( verticesSq, rhs->verticesSq, rhs->nVerticesSq*sizeof(gpuFloatType_t), cudaMemcpyHostToDevice );
		} else {
			// device to host
//			MonteRayMemcpy( vertices, rhs->vertices, rhs->nVertices*sizeof(gpuFloatType_t), cudaMemcpyDeviceToHost );
//			MonteRayMemcpy( verticesSq, rhs->verticesSq, rhs->nVerticesSq*sizeof(gpuFloatType_t), cudaMemcpyDeviceToHost );
		}

		nVertices = rhs->nVertices;
		nVerticesSq = rhs->nVerticesSq;
		delta = rhs->delta;
		minVertex = rhs->minVertex;
		maxVertex = rhs->maxVertex;
		numBins = rhs->numBins;
		type = rhs->type;
		radialModified = rhs->radialModified;
#else
		throw std::runtime_error("MonteRay_GridBins::copy -- can NOT copy between host and device without CUDA.");
#endif
	}

    void initialize( gpuFloatType_t min, gpuFloatType_t max, unsigned nBins);

    void initialize( const std::vector<gpuFloatType_t>& bins );

    void setup(void);

    CUDA_CALLABLE_MEMBER unsigned getNumBins(void) const {
    	//if( debug ) printf("Debug: MonteRay_GridBins::getNumBins -- \n");
    	return numBins;
    }
    CUDA_CALLABLE_MEMBER gpuFloatType_t getMinVertex(void) const {return minVertex; }
    CUDA_CALLABLE_MEMBER gpuFloatType_t getMaxVertex(void) const {return maxVertex; }
    CUDA_CALLABLE_MEMBER unsigned getNumVertices(void) const { return nVertices; }

    CUDA_CALLABLE_MEMBER gpuFloatType_t getDelta(void) const { return delta; }

    CUDA_CALLABLE_MEMBER const gpuFloatType_t* getVerticesData(void) const {return vertices;}

    void removeVertex(unsigned i);

    void modifyForRadial(void);

    CUDA_CALLABLE_MEMBER bool isLinear(void) const { if( type == LINEAR) return true; return false; }
    CUDA_CALLABLE_MEMBER bool isRadial(void) const { if( type == RADIAL) return true; return false; }

    // returns -1 for one neg side of mesh
    // and number of bins on the pos side of the mesh
    CUDA_CALLABLE_MEMBER int getLinearIndex(gpuFloatType_t pos) const;

    CUDA_CALLABLE_MEMBER int getRadialIndexFromR( gpuFloatType_t r) const { MONTERAY_ASSERT( r >= 0.0 ); return getRadialIndexFromRSq( r*r ); }
    CUDA_CALLABLE_MEMBER int getRadialIndexFromRSq( gpuFloatType_t rSq) const;

    CUDA_CALLABLE_MEMBER bool isIndexOutside( int i) const { if( i < 0 ||  i >= getNumBins() ) return true; return false; }

public:

    unsigned nVertices;
    unsigned nVerticesSq;

    std::vector<gpuFloatType_t>* verticesVec = nullptr;
    std::vector<gpuFloatType_t>* verticesSqVec = nullptr;

    gpuFloatType_t* vertices;
    gpuFloatType_t* verticesSq;

    gpuFloatType_t delta;

private:
    gpuFloatType_t minVertex;
    gpuFloatType_t maxVertex;
    unsigned numBins;
    coordinate_t type;
    bool radialModified;

    void validate();

    const bool debug = false;

public:
    void write( const std::string& fileName );
    void read( const std::string& fileName );

    void write(std::ostream& outfile) const;

    void read(std::istream& infile);
    void read_v0(std::istream& infile);

};

} /* namespace MonteRay */

#endif /* MonteRay_GridBins_HH_ */
