#include "MonteRayTally.hh"

#include <mpi.h>

#include "MonteRayMemory.hh"
#include "GPUAtomicAdd.hh"
#include "MonteRayCopyMemory.t.hh"
#include "MonteRayParallelAssistant.hh"

namespace MonteRay{

MonteRayTally::~MonteRayTally(){
    if( Base::isCudaIntermediate ) {
        if( pTimeBinElements ) MonteRayDeviceFree( pTimeBinElements );
        if( pData ) MonteRayDeviceFree( pData );
    } else {
        if( pTimeBinElements ) delete pTimeBinEdges;
        if( pData ) MonteRayHostFree( pData, Base::isManagedMemory );
    }
}

// required for CopyMemoryBase
void
MonteRayTally::copy(const MonteRayTally* rhs) {
#ifdef __CUDACC__
    if( MonteRayParallelAssistant::getInstance().getWorkGroupRank() != 0 ) return;

    if( isCudaIntermediate and !rhs->initialized ) {
        throw std::runtime_error( "MonteRayTally::copy -- tally not initialized!!");
    }

    if( debug ) {
        std::cout << "Debug: MonteRayTally::copy(const MonteRayTally* rhs) \n";
    }

    if( isCudaIntermediate && rhs->isCudaIntermediate ) {
        throw std::runtime_error("MonteRayTally::copy -- can NOT copy CUDA intermediate to CUDA intermediate.");
    }

    if( !isCudaIntermediate && !rhs->isCudaIntermediate ) {
        throw std::runtime_error("RayList_t::copy -- can NOT copy CUDA non-intermediate to CUDA non-intermediate.");
    }

    if( isCudaIntermediate ) {
        // host to device
        if( numTimeBinElements == 0 ) {
            pTimeBinElements = (gpuFloatType_t*) MONTERAYDEVICEALLOC( rhs->numTimeBinElements*sizeof(gpuFloatType_t), std::string("device - MonteRayTally::pTimeBinElements") );
        }
        MonteRayMemcpy( pTimeBinElements, rhs->pTimeBinElements, rhs->numTimeBinElements*sizeof(gpuFloatType_t), cudaMemcpyHostToDevice );

        if( data_size == 0 ) {
            pData = (gpuTallyType_t*) MONTERAYDEVICEALLOC( (rhs->data_size)*sizeof(gpuTallyType_t), std::string("device - MonteRayTally::pData") );
            if( debug ) printf("Debug: MonteRayTally::copy -- allocated pData on the device ptr=%p, size = %d\n",pData, rhs->data_size);
        }
        MonteRayMemcpy( pData, rhs->pData, rhs->data_size*sizeof(gpuTallyType_t), cudaMemcpyHostToDevice );

        data_size = rhs->data_size;
        numTimeBinElements = rhs->numTimeBinElements;
        numSpatialBins = rhs->numSpatialBins;
    } else {
        // device to host
        if( debug ) printf("Debug: MonteRayTally::copy -- device to host , host pData=%p, device pData ptr=%p, nElements=%d\n", pData, rhs->pData, data_size);

        gpuTallyType_t* pTempGPUTally =  (gpuTallyType_t*) MONTERAYHOSTALLOC( (data_size)*sizeof( gpuTallyType_t ), isManagedMemory, std::string("MonteRayTally::copy::pTempGPUTally") );
        MonteRayMemcpy( pTempGPUTally, rhs->pData, (data_size)*sizeof(gpuTallyType_t), cudaMemcpyDeviceToHost );
        for( unsigned i=0; i<data_size; ++i ) {
            pData[i] += pTempGPUTally[i];
            pTempGPUTally[i] = 0.0;
        }

        // zero out the GPU tally
        //memset( pTempGPUTally, 0, data_size*sizeof( gpuTallyType_t ) );
        MonteRayMemcpy( rhs->pData, pTempGPUTally, data_size*sizeof(gpuTallyType_t), cudaMemcpyHostToDevice );
        MonteRayHostFree( pTempGPUTally, Base::isManagedMemory );
    }
#endif

}

CUDA_CALLABLE_MEMBER
void
MonteRayTally::scoreByIndex(gpuTallyType_t value, unsigned spatial_index, unsigned time_index ) {
    unsigned index = getIndex( spatial_index, time_index );
    gpu_atomicAdd( &( pData[index]), value);
}

CUDA_CALLABLE_MEMBER
void
MonteRayTally::score(gpuTallyType_t value, unsigned spatial_index, gpuFloatType_t time  ) {
    unsigned time_index = getTimeIndex( time );
    scoreByIndex( value, spatial_index, time_index );
}


void
MonteRayTally::setupForParallel() {}

void
MonteRayTally::gather() {
    copyToCPU();
    if( ! MonteRayParallelAssistant::getInstance().isParallel() ) return;

    gpuTallyType_t* pGlobalData;

    if( MonteRayParallelAssistant::getInstance().getInterWorkGroupRank() == 0 ) {
        pGlobalData =  (gpuTallyType_t*) MONTERAYHOSTALLOC( (data_size)*sizeof( gpuTallyType_t ), isManagedMemory, std::string("MonteRayTally::pGlobalData") );
        memset( pGlobalData, 0.0, data_size);
    }

    if( MonteRayParallelAssistant::getInstance().getInterWorkGroupCommunicator() != MPI_COMM_NULL ) {
        MPI_Reduce( pData, pGlobalData, data_size, MPI_DOUBLE, MPI_SUM, 0, MonteRayParallelAssistant::getInstance().getInterWorkGroupCommunicator());
    }

    if( MonteRayParallelAssistant::getInstance().getInterWorkGroupRank() != 0 ) {
        memset( pData, 0, data_size*sizeof( gpuTallyType_t ) );
    } else {
        MonteRayHostFree( pData, Base::isManagedMemory );
        pData = pGlobalData;
    }
}

void
MonteRayTally::gatherWorkGroup() {
    // For testing - setup like gather but allows direct scoring on all ranks of the work group
    if( ! MonteRayParallelAssistant::getInstance().isParallel() ) return;
    copyToCPU();

    gpuTallyType_t* pGlobalData;

    if( MonteRayParallelAssistant::getInstance().getWorkGroupRank() == 0 ) {
        pGlobalData =  (gpuTallyType_t*) MONTERAYHOSTALLOC( (data_size)*sizeof( gpuTallyType_t ), isManagedMemory, std::string("MonteRayTally::pGlobalData") );
        memset( pGlobalData, 0.0, data_size);
    }

    if( MonteRayParallelAssistant::getInstance().getWorkGroupCommunicator() != MPI_COMM_NULL ) {
        MPI_Reduce( pData, pGlobalData, data_size, MPI_DOUBLE, MPI_SUM, 0, MonteRayParallelAssistant::getInstance().getWorkGroupCommunicator());
    }

    if( MonteRayParallelAssistant::getInstance().getWorkGroupRank() != 0 ) {
        memset( pData, 0, data_size*sizeof( gpuTallyType_t ) );
    } else {
        MonteRayHostFree( pData, Base::isManagedMemory );
        pData = pGlobalData;
    }
}

template class CopyMemoryBase<MonteRayTally>;

}
