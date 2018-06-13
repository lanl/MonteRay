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

#include "MonteRayDefinitions.hh"
#include "GPUErrorCheck.hh"

namespace MonteRay {

/// MonteRay GPU Properties class
class MonteRayGPUProps {
public:
	MonteRayGPUProps() :
		deviceID( -1 ),
		numDevices( -1 )
	{
#ifdef __CUDACC__
		deviceProps = cudaDevicePropDontCare;
		cudaGetDeviceCount(&numDevices);

		if( numDevices <= 0 ) {
			throw std::runtime_error("MonteRayGPUProps::MonteRayGPUProps() -- No GPU found.");
		}

		// TODO: setup for multiple devices
		cudaGetDevice(&deviceID);

		cudaGetDeviceProperties(&deviceProps , deviceID);

		if( ! deviceProps.managedMemory ) {
			std::cout << "MONTERAY WARNING: GPU does not support managed memory.\n";
		}
		if( ! deviceProps.concurrentManagedAccess ) {
			std::cout << "MONTERAY WARNING: GPU does not support concurrent managed memory access.\n";
		}
	#else
		throw std::runtime_error("MonteRayGPUProps::MonteRayGPUProps() -- CUDA not enabled.");
	#endif
	}

	MonteRayGPUProps( const MonteRayGPUProps& rhs ) {
		deviceID = rhs.deviceID;
		numDevices = rhs.numDevices;
#ifdef __CUDACC__
		deviceProps = rhs.deviceProps;
#endif
	}

	~MonteRayGPUProps(){}

	int deviceID; ///< GPU ID Number
	int numDevices;

#ifdef __CUDACC__
	cudaDeviceProp deviceProps;
#endif
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
	allocationInfo_t(){
		name = "";
		size = 0;
		ptr = NULL;
		allocated = false;
		hostmemory = true;
	}
	std::string name;
	size_t size;
	void* ptr;
	bool allocated;
	bool hostmemory;
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
				printf( "Debug: Allocation ID = %d not freed, ptr address = %p, size = %d bytes, Name = %s\n",
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

inline void* MonteRayHostAlloc(size_t len, bool managed = true, std::string name = std::string(""), const char *file = "", int line = -1 ) {
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

#define MONTERAYHOSTALLOC(len, managed, name)({ MonteRay::MonteRayHostAlloc(len, managed, name, __FILE__, __LINE__); })

inline void* MonteRayDeviceAlloc(size_t len, std::string name = std::string(""), const char *file = "", int line = -1 ){
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

#define MONTERAYDEVICEALLOC(len, name)({ MonteRay::MonteRayDeviceAlloc(len, name, __FILE__, __LINE__); })

inline void MonteRayHostFree(void* ptr, bool managed ) noexcept {
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

inline void MonteRayDeviceFree(void* ptr) noexcept {
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
	throw std::runtime_error( "MonteRayDeviceFree -- can NOT free device memory without CUDA." );
#endif
}

#ifdef __CUDACC__
inline void MonteRayMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind){
	CUDA_CHECK_RETURN( cudaMemcpy(dst, src, count, kind) );
}
#endif

} // end namespace



#endif /* MONTERAYMEMORY_HH_ */
