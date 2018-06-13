#ifndef HASHBINS_H_
#define HASHBINS_H_

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"
#include "BinarySearch.hh"
#include "MonteRayCopyMemory.hh"

namespace MonteRay{

class HashBins : public CopyMemoryBase<HashBins> {
private:
    gpuFloatType_t min = 0.0;
    gpuFloatType_t max = 0.0 ;
    gpuFloatType_t invDelta = 0.0;
    unsigned nEdges = 0;
    unsigned* binBounds = nullptr;

public:
    using Base = MonteRay::CopyMemoryBase<HashBins> ;

	HashBins(gpuFloatType_t *vertices, unsigned nVertices, unsigned nHashBinEdges = 8000) : CopyMemoryBase() {
		MONTERAY_ASSERT( nHashBinEdges > 1 );

		nEdges = nHashBinEdges;

		binBounds = (unsigned*) MONTERAYHOSTALLOC( nEdges * sizeof( unsigned ), false, std::string("HashBins::binBounds") );

		min = vertices[0];
		max = vertices[nVertices-1];

		gpuFloatType_t delta = (max-min)/(nEdges-1);
		invDelta = 1/delta;

		for( unsigned i=0; i< nEdges; ++i ) {
			double edge = getBinEdge(i);
			binBounds[i] = LowerBoundIndex( vertices, nVertices, edge);
		}
	}

	~HashBins() {
		if( Base::isCudaIntermediate ) {
			MonteRayDeviceFree( binBounds );
		} else {
			MonteRayHostFree( binBounds, Base::isManagedMemory );
		}
	}

	std::string className(){ return std::string("HashBins");}

	void init() {
		min = 0.0;
		max = 0.0;
		invDelta = 0.0;
		nEdges = 0;
		binBounds = nullptr;
	}

	void copy(const HashBins* rhs) {
#ifdef __CUDACC__
		if( nEdges != 0 && (nEdges != rhs->nEdges) ) {
			std::cout << "Error: HashBins::copy -- can't change size after initialization.\n";
			std::cout << "Error: HashBins::copy -- nEdges = " << nEdges << " \n";
			std::cout << "Error: HashBins::copy -- rhs->nEdges = " << rhs->nEdges << " \n";
			std::cout << "Error: HashBins::copy -- isCudaIntermediate = " << isCudaIntermediate << " \n";
			std::cout << "Error: HashBins::copy -- rhs->isCudaIntermediate = " << rhs->isCudaIntermediate << " \n";
			throw std::runtime_error("HashBins::copy -- can't change size after initialization.");
		}

		if( isCudaIntermediate ) {
			// host to device
			if( nEdges == 0 ) {
				binBounds = (unsigned*) MONTERAYDEVICEALLOC( rhs->nEdges*sizeof(unsigned), std::string("device - HashBins::binBounds") );
			}
			MonteRayMemcpy( binBounds, rhs->binBounds, rhs->nEdges*sizeof(gpuFloatType_t), cudaMemcpyHostToDevice );
		} else {
			// device to host
			MonteRayMemcpy( binBounds, rhs->binBounds, rhs->nEdges*sizeof(gpuFloatType_t), cudaMemcpyDeviceToHost );
		}

		min = rhs->min;
		max = rhs->max;
		invDelta = rhs->invDelta;
		nEdges = rhs->nEdges;
#else
		throw std::runtime_error("HashBins::copy -- can NOT copy between host and device without CUDA.");
#endif
	}

	CUDA_CALLABLE_MEMBER
	gpuFloatType_t getMin(void) const { return min; }

	CUDA_CALLABLE_MEMBER
	gpuFloatType_t getMax(void) const { return max; }

	CUDA_CALLABLE_MEMBER
	gpuFloatType_t getDelta(void) const { return 1/invDelta; }

	CUDA_CALLABLE_MEMBER
	gpuFloatType_t getBinEdge(unsigned i) const { return (1/invDelta)*i+min; }

	CUDA_CALLABLE_MEMBER
	unsigned getBinBound( unsigned i ) const { return binBounds[i]; }

	CUDA_CALLABLE_MEMBER
	unsigned getNEdges( void ) const { return nEdges; }

	CUDA_CALLABLE_MEMBER
	void getLowerUpperBins( double value, unsigned& lower, unsigned& upper) const {
		MONTERAY_ASSERT( value > min );
		MONTERAY_ASSERT( value < max );
		unsigned bin = (value-min) * invDelta;
		lower = binBounds[bin];
		upper = binBounds[bin+1];
	}

};


}
#endif /* HASHBINS_H_ */
