#include "RayWorkInfo.hh"
#include "MonteRayCopyMemory.t.hh"

namespace MonteRay {

CUDAHOST_CALLABLE_MEMBER
RayWorkInfo::RayWorkInfo(unsigned num, bool cpuAllocate) {
    if( num == 0 ) { num = 1; }
    if( num > MONTERAY_MAX_THREADS ) {
        std::cout << "WARNING: Limiting MonteRay RayWorkInfo size, requested size="
                  << num << ", limit=" << MONTERAY_MAX_THREADS << "\n";
        num = MONTERAY_MAX_THREADS;
    }

#ifdef DEBUG
    if( Base::debug ) {
        std::cout << "RayWorkInfo::RayWorkInfo(n), n=" << num << " \n";
    }
#endif
    allocateOnCPU = cpuAllocate;

    reallocate( num );
}

CUDAHOST_CALLABLE_MEMBER
void
RayWorkInfo::reallocate(unsigned n) {
    if( indices          != NULL ) { MonteRayHostFree( indices,          Base::isManagedMemory ); }
    if( rayCastSize      != NULL ) { MonteRayHostFree( rayCastSize,      Base::isManagedMemory ); }
    if( rayCastCell      != NULL ) { MonteRayHostFree( rayCastCell,      Base::isManagedMemory ); }
    if( rayCastDistance  != NULL ) { MonteRayHostFree( rayCastDistance,  Base::isManagedMemory ); }
    if( crossingSize     != NULL ) { MonteRayHostFree( crossingSize,     Base::isManagedMemory ); }
    if( crossingCell     != NULL ) { MonteRayHostFree( crossingCell,     Base::isManagedMemory ); }
    if( crossingDistance != NULL ) { MonteRayHostFree( crossingDistance, Base::isManagedMemory ); }

    init();

    if( allocateOnCPU ) {
        // only allocate if not using CUDA -- very big
        indices = (int*) MONTERAYHOSTALLOC( n*3*sizeof( int ), Base::isManagedMemory, "host RayWorkInfo::indices" );
        rayCastSize = (int*) MONTERAYHOSTALLOC( n*sizeof( int ), Base::isManagedMemory, "host RayWorkInfo::rayCastSize" );
        rayCastCell = (int*) MONTERAYHOSTALLOC( n*MAXNUMRAYCELLS*sizeof( int ), Base::isManagedMemory, "host RayWorkInfo::rayCastCell" );
        rayCastDistance = (gpuRayFloat_t*) MONTERAYHOSTALLOC( n*MAXNUMRAYCELLS*sizeof( gpuRayFloat_t ), Base::isManagedMemory, "host RayWorkInfo::rayCastDistance" );
        crossingSize = (int*) MONTERAYHOSTALLOC( n*3*sizeof( int ), Base::isManagedMemory, "host RayWorkInfo::crossingSize" );
        crossingCell = (int*) MONTERAYHOSTALLOC( n*MAXNUMVERTICES*3*sizeof( int ), Base::isManagedMemory, "host RayWorkInfo::crossingCell" );
        crossingDistance = (gpuRayFloat_t*) MONTERAYHOSTALLOC( n*MAXNUMVERTICES*3*sizeof( gpuRayFloat_t ), Base::isManagedMemory, "host RayWorkInfo::crossingDistance" );
    }

    nAllocated = n;
    clear();
}

CUDAHOST_CALLABLE_MEMBER
RayWorkInfo::~RayWorkInfo(){
    if( ! Base::isCudaIntermediate ) {
        MonteRayHostFree(indices, Base::isManagedMemory );
        MonteRayHostFree(rayCastSize, Base::isManagedMemory );
        MonteRayHostFree(rayCastCell, Base::isManagedMemory );
        MonteRayHostFree(rayCastDistance, Base::isManagedMemory );
        MonteRayHostFree(crossingSize, Base::isManagedMemory );
        MonteRayHostFree(crossingCell, Base::isManagedMemory );
        MonteRayHostFree(crossingDistance, Base::isManagedMemory );
    } else {
        MonteRayDeviceFree( indices );
        MonteRayDeviceFree( rayCastSize );
        MonteRayDeviceFree( rayCastCell );
        MonteRayDeviceFree( rayCastDistance );
        MonteRayDeviceFree( crossingSize );
        MonteRayDeviceFree( crossingCell );
        MonteRayDeviceFree( crossingDistance );
    }
}

CUDAHOST_CALLABLE_MEMBER void
RayWorkInfo::copy(const RayWorkInfo* rhs) {

#ifdef __CUDACC__

#ifdef DEBUG
    if( Base::debug ) {
        std::cout << "Debug: RayWorkInfo::copy (const RayWorkInfo& rhs) \n";
    }
#endif

    if( Base::isCudaIntermediate && rhs->isCudaIntermediate ) {
        throw std::runtime_error("RayWorkInfo::copy -- can NOT copy CUDA intermediate to CUDA intermediate.");
    }

    if( !Base::isCudaIntermediate && !rhs->isCudaIntermediate ) {
        throw std::runtime_error("RayWorkInfo::copy -- can NOT copy CUDA non-intermediate to CUDA non-intermediate.");
    }

    if( nAllocated > 0 && nAllocated != rhs->nAllocated) {
        throw std::runtime_error("RayWorkInfo::copy -- can NOT change the size of the RayList.");
    }

    if( Base::isCudaIntermediate ) {
        // target is the intermediate, origin is the host
        if( indices == NULL ) {
            indices = (int*) MONTERAYDEVICEALLOC( rhs->nAllocated*sizeof(int)*3, "device - RayWorkInfo::indices" );
        }

        if( rayCastSize == NULL ) {
            rayCastSize = (int*) MONTERAYDEVICEALLOC( rhs->nAllocated*sizeof(int), "device - RayWorkInfo::rayCastSize" );
        }
        //cudaMemset(rayCastSize, 0, rhs->nAllocated*sizeof(int));

        if( rayCastCell == NULL ) {
            rayCastCell = (int*) MONTERAYDEVICEALLOC( rhs->nAllocated*sizeof(int)*MAXNUMRAYCELLS, "device - RayWorkInfo::rayCastCell" );
        }

        if( rayCastDistance == NULL ) {
            rayCastDistance = (gpuRayFloat_t*) MONTERAYDEVICEALLOC( rhs->nAllocated*sizeof(gpuRayFloat_t)*MAXNUMRAYCELLS, "device - RayWorkInfo::rayCastDistance" );
        }

        if( crossingSize == NULL ) {
            crossingSize = (int*) MONTERAYDEVICEALLOC( rhs->nAllocated*sizeof(int)*3, "device - RayWorkInfo::crossingSize" );
        }
        //cudaMemset(crossingSize, 0, rhs->nAllocated*3*sizeof(int));

        if( crossingCell == NULL ) {
            crossingCell = (int*) MONTERAYDEVICEALLOC( rhs->nAllocated*sizeof(int)*MAXNUMVERTICES*3, "device - RayWorkInfo::crossingCell" );
        }

        if( crossingDistance == NULL ) {
            crossingDistance = (gpuRayFloat_t*) MONTERAYDEVICEALLOC( rhs->nAllocated*sizeof(gpuRayFloat_t)*MAXNUMVERTICES*3, "device - RayWorkInfo::crossingDistance" );
        }

        // initialize the crossing info
//        MonteRayMemcpy(rayCastSize, rhs->rayCastSize, rhs->nAllocated*sizeof(int), cudaMemcpyHostToDevice);
//        MonteRayMemcpy(crossingSize, rhs->crossingSize, rhs->nAllocated*sizeof(int)*3, cudaMemcpyHostToDevice);
    }else {
        // device to host
        if( allocateOnCPU ) {
            MonteRayMemcpy( indices, rhs->indices, rhs->nAllocated*sizeof(int)*3, cudaMemcpyDeviceToHost );
            MonteRayMemcpy( rayCastSize, rhs->rayCastSize, rhs->nAllocated*sizeof(int), cudaMemcpyDeviceToHost );
            MonteRayMemcpy( rayCastCell, rhs->rayCastCell, rhs->nAllocated*sizeof(int)*MAXNUMRAYCELLS, cudaMemcpyDeviceToHost );
            MonteRayMemcpy( rayCastDistance, rhs->rayCastDistance, rhs->nAllocated*sizeof(gpuRayFloat_t)*MAXNUMRAYCELLS, cudaMemcpyDeviceToHost );
            MonteRayMemcpy( crossingSize, rhs->crossingSize, rhs->nAllocated*sizeof(int)*3, cudaMemcpyDeviceToHost );
            MonteRayMemcpy( crossingCell, rhs->crossingCell, rhs->nAllocated*sizeof(int)*MAXNUMVERTICES*3, cudaMemcpyDeviceToHost );
            MonteRayMemcpy( crossingDistance, rhs->crossingDistance, rhs->nAllocated*sizeof(gpuRayFloat_t)*MAXNUMVERTICES*3, cudaMemcpyDeviceToHost );
        }
    }

    nAllocated = rhs->nAllocated;
#else
    throw std::runtime_error("RayWorkInfo::copy -- Only valid when compiling with CUDA.");
#endif
}

CUDAHOST_CALLABLE_MEMBER void
RayWorkInfo::addRayCastCell(unsigned i, int cellID, gpuRayFloat_t dist) {
#ifdef DEBUG
    if( dist < 0.0 ) {
        printf("Debug:  ERROR: RayWorkInfo::addRayCastCell, distance < 0, threadID=%u, cellID=%d, distance=%f\n", i, cellID, dist );
    }
#endif
    MONTERAY_ASSERT_MSG( dist >= 0, "distance must be > 0.0!" );
    getRayCastCell(i, getRayCastSize(i) ) = cellID;
    getRayCastDist(i, getRayCastSize(i) ) = dist;
    ++(getRayCastSize(i));
}

CUDAHOST_CALLABLE_MEMBER void
RayWorkInfo::addCrossingCell(unsigned dim, unsigned i, int cellID, gpuRayFloat_t dist) {
#ifdef DEBUG
    if( dist < 0.0 ) {
        printf("Debug:  ERROR: RayWorkInfo::addCrossingCell, distance < 0, threadID=%u, cellID=%d, distance=%f\n", i, cellID, dist );
    }
#endif
    MONTERAY_ASSERT_MSG( dist >= 0, "distance must be > 0.0!" );
    getCrossingCell(dim, i, getCrossingSize(dim,i) ) = cellID;
    getCrossingDist(dim, i, getCrossingSize(dim,i) ) = dist;
    ++(getCrossingSize(dim,i));
}

template class CopyMemoryBase<RayWorkInfo>;

} // end namespace
