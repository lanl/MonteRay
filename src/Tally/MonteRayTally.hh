#ifndef MONTERAYTALLY_HH_
#define MONTERAYTALLY_HH_

#include <vector>
#include <string>
#include <memory>

#include "MonteRayCopyMemory.hh"
#include "MonteRayConstants.hh"
#include "MonteRayParallelAssistant.hh"

namespace MonteRay {

class MonteRayTally : public CopyMemoryBase<MonteRayTally> {
public:
    using Base = MonteRay::CopyMemoryBase<MonteRayTally> ;

private:
    bool copiedToGPU = false;
    bool initialized = false;

    unsigned numSpatialBins = 1;

    std::vector<gpuFloatType_t>* pTimeBinEdges;
    gpuFloatType_t* pTimeBinElements = NULL;
    unsigned numTimeBinElements = 0;

    unsigned data_size = 0;
    gpuTallyType_t* pData = NULL;

public:
    MonteRayTally(unsigned nSpatialBins=1) :
        CopyMemoryBase<MonteRayTally>()
    {
        init();
        if( nSpatialBins == 0 ) {
            nSpatialBins = 1;
        }
        numSpatialBins =  nSpatialBins;
    }

    ~MonteRayTally();

    // required for CopyMemoryBase
    void init() {
        copiedToGPU = false;
        initialized = false;

        numSpatialBins = 1;
        if( ! Base::isCudaIntermediate ) {
            pTimeBinEdges = new std::vector<gpuFloatType_t>;
        }
        pTimeBinElements = NULL;
        pData = NULL;
        numTimeBinElements = 0;
        data_size = 0;
    }

    void initialize() {
        if( initialized ) {
            throw std::runtime_error("Error: MonteRay::MonteRayTally::initialize() -- already initialized!!");
        }
        initialized = true;

        if( numTimeBinElements == 0 ) {
            std::vector<gpuFloatType_t> default_edges = { float_inf };
            setTimeBinEdges( default_edges );
        }
        data_size = getIndex( (numSpatialBins-1), numTimeBinElements );
//        std::cout << "Debug:  MonteRayTally::initialize() -- numTimeBinElements=" << numTimeBinElements << "\n";
//        std::cout << "Debug:  MonteRayTally::initialize() -- numSpatialBins=" << numSpatialBins << "\n";
//        std::cout << "Debug:  MonteRayTally::initialize() -- data_size=" << data_size << "\n";

        if( ! Base::isCudaIntermediate ) {
            pData = (gpuTallyType_t*) MONTERAYHOSTALLOC( (data_size)*sizeof( gpuTallyType_t ), isManagedMemory, std::string("MonteRayTally::pData") );

#ifdef DEBUG
            if( debug ) printf("Debug: MonteRayTally::copy -- allocated pData on the host ptr=%p, size = %d\n",pData, data_size);
#endif

        }

        memset( pData, 0, data_size*sizeof( gpuTallyType_t ) );
    }

    void setupForParallel();

    void gather();

    // Used mainly for testing
    void gatherWorkGroup();

    // required for CopyMemoryBase
    void copy(const MonteRayTally* rhs);

    void copyToGPU(void) {
        if( MonteRayParallelAssistant::getInstance().getWorkGroupRank() != 0 ) return;
        copiedToGPU = true;

        //std::cout << "Debug: MonteRayTally::copyToGPU \n";
        Base::copyToGPU();
    }

    void copyToCPU(void) {
        if( MonteRayParallelAssistant::getInstance().getWorkGroupRank() != 0 ) return;
        if( !copiedToGPU ) return;

        //if( debug ) std::cout << "Debug: MonteRayTally::copyToCPU \n";
        Base::copyToCPU();
    }

    // required for CopyMemoryBase
    std::string className() { return std::string("MonteRayTally"); }

    bool isInitialized() const { return initialized; }

    CUDA_CALLABLE_MEMBER unsigned getNumSpatialBins() const {
        return numSpatialBins;
    }

    template<typename T>
    void setTimeBinEdges( std::vector<T> edges) {
        if( edges.size() == 0 ) { edges.push_back( float_inf ); }
        if( edges.front() == 0.0 ) { edges.erase( edges.begin() ); }
        if( edges.back() < float_inf ) { edges.push_back( float_inf ); }
        pTimeBinEdges->resize( edges.size() );
        for( unsigned i=0; i<edges.size(); ++i) {
            (*pTimeBinEdges)[i] = edges[i];
        }

        pTimeBinElements = pTimeBinEdges->data();
        numTimeBinElements = pTimeBinEdges->size();
    }

    CUDA_CALLABLE_MEMBER
    unsigned getNumTimeBins() const {
        return numTimeBinElements;
    }

    gpuFloatType_t getTimeBinEdge(unsigned i ) const {
        if( pTimeBinEdges ) {
            return pTimeBinEdges->at(i);
        }
        return 0.0;
    }

    CUDA_CALLABLE_MEMBER
    unsigned getIndex(unsigned spatial_index, unsigned time_index) const {
        // layout is first by spatial index, then by time index
        // assuming threads are contributing to the same time index
        return  time_index*numSpatialBins + spatial_index;
    }

    CUDA_CALLABLE_MEMBER
    unsigned getTimeIndex( const gpuFloatType_t time ) const {

        // TODO: test linear search (done here) vs. binary search
        unsigned index = numTimeBinElements-1;
        for( int i = 0; i < numTimeBinElements-1 ; ++i) {
            if( time <  pTimeBinElements[i] ) {
                index = i;
                break;
            }
        }
        return index;
    }

    CUDA_CALLABLE_MEMBER
    void scoreByIndex(gpuTallyType_t value, unsigned spatial_index, unsigned time_index=0 );

    CUDA_CALLABLE_MEMBER
    void score( gpuTallyType_t value, unsigned spatial_index, gpuFloatType_t time = 0.0 );

    CUDA_CALLABLE_MEMBER
    gpuTallyType_t getTally(unsigned spatial_index, unsigned time_index = 0 ) const {
        if( pData ) {
            return pData[ getIndex(spatial_index, time_index) ];
        } else {
            return 0.0;
        }
    }

    CUDA_CALLABLE_MEMBER
    unsigned getTallySize() const {
        return data_size;
    }

};

} // end namespace

#endif /* MONTERAYTALLY_HH_ */
