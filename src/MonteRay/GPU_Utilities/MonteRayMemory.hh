#ifndef MONTERAYMEMORY_HH_
#define MONTERAYMEMORY_HH_

#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <typeinfo>
#include <iostream>
#include <atomic>
#include <vector>
#include <mutex>

class cudaDeviceProp;

namespace MonteRay {

/// MonteRay GPU Properties class
class MonteRayGPUProps {
public:
    MonteRayGPUProps();

    MonteRayGPUProps( const MonteRayGPUProps& rhs );

    ~MonteRayGPUProps();

    int deviceID = -1; ///< GPU ID Number
    int numDevices = -1;

    cudaDeviceProp* deviceProps = nullptr;
};

typedef size_t alloc_id_t;

#define TRACKALLOCATIONS 1

#if TRACKALLOCATIONS > 0
static const bool trackAllocations = true;
static const size_t id_offset = sizeof( alloc_id_t );
#else
static const bool trackAllocations = false;
static const size_t id_offset = 0;
#endif

static const bool debugAllocations = false;

class allocationInfo_t {
public:
    allocationInfo_t(){}

    std::string name;
    size_t size = 0;
    void* ptr = NULL;
    bool allocated = false;
    bool hostmemory = true;
};

static std::mutex allocationMutex;

class AllocationTracker {
public:
    typedef size_t alloc_id_t;

    static AllocationTracker& getInstance() {
        static AllocationTracker instance;
        return instance;
    }
    ~AllocationTracker() { reportLeakedMemory(); }

    void reportLeakedMemory(void) {
        std::lock_guard<std::mutex> lock(allocationMutex);

        if( allocationList.size() == 0 ) return;
        printf("*****************************************************\n");
        printf("************ MonteRay Memory Report *****************\n\n");
        printf(" Maximum memory used on the host   = %12.6f MB\n", (float) maxCPUMemory / 1000000.0 );
        printf(" Maximum memory used on the device = %12.6f MB\n", (float) maxGPUMemory / 1000000.0 );
        printf(" Total leaked memory on the host   = %12.6f MB\n", (float) currentCPUMemory / 1000000.0 );
        printf(" Total leaked memory on the device = %12.6f MB\n", (float) currentGPUMemory / 1000000.0 );
        for( unsigned i = 0; i < allocationList.size(); ++i ) {
            if( allocationList[i].allocated == true ) {
                alloc_id_t id = i;
                std::string name = allocationList[i].name;
                size_t size =allocationList[i].size;
                void* ptr = allocationList[i].ptr;
                printf( "Debug: Allocation ID = %lu not freed, ptr address = %p, size = %lu bytes, Name = %s\n",
                        id, ptr, size, name.c_str() );
            }
        }
        printf("*****************************************************\n");
    }
private:
    AllocationTracker()
// : allocationMutex()
{
        std::lock_guard<std::mutex> lock(allocationMutex);
        allocationList.reserve(100);
        currentCPUMemory = 0;
        maxCPUMemory = 0;
        currentGPUMemory = 0;
        maxGPUMemory = 0;
}
public:
    AllocationTracker( AllocationTracker const & ) = delete;
    void operator=( AllocationTracker const & )    = delete;

    alloc_id_t increment(std::string name, void* ptr, size_t numBytes, bool cpuMemory) {

        allocationInfo_t newAllocation;
        newAllocation.name = name;
        newAllocation.size = numBytes;
        newAllocation.ptr = ptr;
        newAllocation.allocated = true;
        newAllocation.hostmemory = cpuMemory;

        if( cpuMemory ) {
            newAllocation.hostmemory = true;
            currentCPUMemory += numBytes;
            if( currentCPUMemory > maxCPUMemory ) {
                maxCPUMemory = currentCPUMemory;
            }
        } else {
            newAllocation.hostmemory = false;
            currentGPUMemory += numBytes;
            if( currentGPUMemory > maxGPUMemory ) {
                maxGPUMemory = currentGPUMemory;
            }
        }

        std::lock_guard<std::mutex> lock(allocationMutex);
        allocationList.push_back( newAllocation );
        alloc_id_t id = allocationList.size()-1;

        return id;
    }

    void decrement( alloc_id_t id ) {
        std::lock_guard<std::mutex> lock(allocationMutex);
        size_t len = allocationList[id].size;
        allocationList[id].allocated = false;
        if( allocationList[id].hostmemory ) {
            currentCPUMemory -= len;
        } else {
            currentGPUMemory -= len;
        }
    }

public:
    std::vector<allocationInfo_t> allocationList;
    std::atomic<size_t> currentCPUMemory;
    size_t maxCPUMemory;
    std::atomic<size_t> currentGPUMemory;
    size_t maxGPUMemory;
};

void* MonteRayHostAlloc(size_t len, bool managed = true, std::string name = std::string(""), const char *file = "", int line = -1 );

#define MONTERAYHOSTALLOC(len, managed, name)({ MonteRay::MonteRayHostAlloc(len, managed, name, __FILE__, __LINE__); })

void* MonteRayDeviceAlloc(size_t len, std::string name = std::string(""), const char *file = "", int line = -1 );

#define MONTERAYDEVICEALLOC(len, name)({ MonteRay::MonteRayDeviceAlloc(len, name, __FILE__, __LINE__); })

void MonteRayHostFree(void* ptr, bool managed ) noexcept;

void MonteRayDeviceFree(void* ptr) noexcept;

void MonteRayMemcpy(void *dst, const void *src, size_t count, int kind);

} // end namespace



#endif /* MONTERAYMEMORY_HH_ */
