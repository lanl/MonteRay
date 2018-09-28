#include "MonteRayMemory.hh"

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"

namespace MonteRay {

MonteRayGPUProps::MonteRayGPUProps() {
#ifdef __CUDACC__
    deviceProps = new cudaDeviceProp;

    *deviceProps = cudaDevicePropDontCare;
    cudaGetDeviceCount(&numDevices);

    if( numDevices <= 0 ) {
        throw std::runtime_error("MonteRayGPUProps::MonteRayGPUProps() -- No GPU found.");
    }

    // TODO: setup for multiple devices
    cudaGetDevice(&deviceID);

    cudaGetDeviceProperties( deviceProps , deviceID);

    if( ! deviceProps->managedMemory ) {
        std::cout << "MONTERAY WARNING: GPU does not support managed memory.\n";
    }
    if( ! deviceProps->concurrentManagedAccess ) {
        std::cout << "MONTERAY WARNING: GPU does not support concurrent managed memory access.\n";
    }
#else
    throw std::runtime_error("MonteRayGPUProps::MonteRayGPUProps() -- CUDA not enabled.");
#endif
}

MonteRayGPUProps::MonteRayGPUProps( const MonteRayGPUProps& rhs ) {
     deviceID = rhs.deviceID;
     numDevices = rhs.numDevices;
#ifdef __CUDACC__
     deviceProps = new cudaDeviceProp;
     *deviceProps = *rhs.deviceProps;
#endif
 }

MonteRayGPUProps::~MonteRayGPUProps(){
#ifdef __CUDACC__
    delete deviceProps;
#endif
}

void* MonteRayHostAlloc(size_t len, bool managed, std::string name, const char *file , int line ) {
    const bool debug = false;

    name += "::" + std::string(file) + "[" + std::to_string(line) + "]";

    void *ptr;
#ifdef __CUDACC__
    if( managed ) {
        if( debug ){
            std::cout << "Debug: MonteRayHostAlloc: allocating " << len << " bytes with cuda managed memory\n";
        }
        cudaMallocManaged(&ptr, len + id_offset);
    } else {
        if( debug ){
            std::cout << "Debug: MonteRayHostAlloc: allocating " << len << " bytes with malloc\n";
        }

        ptr = malloc(len + id_offset );
    }
#else
    ptr = malloc(len + id_offset );
#endif

    if( trackAllocations ) {
        long long id = AllocationTracker::getInstance().increment(name, ptr, len+id_offset, true); // get unique id
        *(long long *)ptr = id; // store id in front of data
        if( debugAllocations ) {
            printf( "Debug: MonteRayHostAlloc   -- Allocation ID = %d, ptr address = %p, size = %d bytes, Name = %s\n",
                    id, ptr, len+id_offset, name.c_str() );
        }
    }
    return (char *) ptr + id_offset; // return pointer to data
}

void* MonteRayDeviceAlloc(size_t len, std::string name, const char *file, int line ){
    const bool debug = false;

    name += "::" + std::string(file) + "[" + std::to_string(line) + "]";

    void *ptr;
#ifdef __CUDACC__
    if( debug ){
        std::cout << "Debug: MonteRayHostAlloc: allocating " << len << " bytes with cudaMalloc\n";
    }
    cudaMalloc(&ptr, len + id_offset );

    if( trackAllocations ) {
        alloc_id_t id = AllocationTracker::getInstance().increment(name, ptr, len+id_offset,false); // get unique id
        cudaMemcpy(ptr, &id, id_offset, cudaMemcpyHostToDevice); // store id in front of data
        if( debugAllocations ) {
            printf( "Debug: MonteRayDeviceAlloc -- Allocation ID = %d, ptr address = %p, size = %d bytes, Name = %s\n",
                    id, ptr, len+id_offset, name.c_str() );
        }
    }
    return (char *) ptr + id_offset; // return pointer to data
#else
    throw std::runtime_error( "MonteRayDeviceAlloc -- can NOT allocate device memory without CUDA." );
#endif
}

void MonteRayHostFree(void* ptr, bool managed ) noexcept {
    if ( ptr == NULL ) { return; }

    void* realPtr = (char *)ptr - id_offset; // real pointer

    if( trackAllocations ) {
        alloc_id_t id =  *((long long*) realPtr);
        AllocationTracker::getInstance().decrement(id);

        if( debugAllocations ) {
            printf( "Debug: MonteRayHostFree -- Deallocating ID = %d, ptr address = %p, size = %d bytes, Name = %s\n",
                    id, realPtr,
                    AllocationTracker::getInstance().allocationList[id].size,
                    AllocationTracker::getInstance().allocationList[id].name.c_str() );
        }
    }

#ifdef __CUDACC__
    if( managed ) {
        cudaFree( realPtr );
    } else {
        std::free( realPtr );
    }
#else
    std::free( realPtr );
#endif
}

void MonteRayDeviceFree(void* ptr) noexcept {
#ifdef __CUDACC__
    if( debugAllocations ) {
        printf( "Debug: MonteRayDeviceFree -- Deallocating   ptr address = %p\n");
    }

    if( ptr == NULL ) { return; }
    char* realPtr = (char *)ptr - id_offset; // real pointer
    if( trackAllocations ){
        alloc_id_t id;
        cudaMemcpy(&id, realPtr, id_offset, cudaMemcpyDeviceToHost); // copy id from front of data
        AllocationTracker::getInstance().decrement(id);
        if( debugAllocations ) {
            printf( "Debug: MonteRayDeviceFree -- Deallocating ID = %d, ptr address = %p, size = %d bytes, Name = %s\n",
                    id, realPtr,
                    AllocationTracker::getInstance().allocationList[id].size,
                    AllocationTracker::getInstance().allocationList[id].name.c_str() );
        }
    }

    cudaFree(realPtr);
#else
    //throw std::runtime_error( "MonteRayDeviceFree -- can NOT free device memory without CUDA." );
#endif
}

void MonteRayMemcpy(void *dst, const void *src, size_t count, int kind){
#ifdef __CUDACC__
    CUDA_CHECK_RETURN( cudaMemcpy(dst, src, count, cudaMemcpyKind(kind) ) );
#endif
}

}  // end namespace
