#ifndef IPCOMM_SHAREDRAYLIST_HH_
#define IPCOMM_SHAREDRAYLIST_HH_

#include <functional>
#include <cstring>
#include <array>
#include <vector>
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <atomic>

#include "MonteRayTypes.hh"
#include "MonteRayAssert.hh"
#include "MonteRay_timer.hh"
#include "MonteRayParallelAssistant.hh"

namespace MonteRay {

struct bucket_header_t{
    unsigned size; // current number of collisions in bucket
    std::atomic<bool> done; // bucket is full or process is done with partial fill
};

struct rank_info_t{
    std::atomic<bool> allDone; // rank is done with work.
    unsigned currentBucket;
};

template<typename COLLISION_T>
class SharedRayList {
public:
    typedef gpuFloatType_t float_t;
    typedef std::function<void (const COLLISION_T* particle, unsigned N ) > store_func_t;

    typedef std::function<void (bool final)> controllerFlush_func_t;
    typedef std::function<void ()> controllerClear_func_t;

    enum offset_types { RANKINFO=0, BUCKETHEADER=1, COLLISIONBUFFER=2};

    template<class T>
    SharedRayList(T& controller, unsigned size, unsigned numBuckets=1000 ) :
        PA( MonteRayParallelAssistant::getInstance() )
    {
        unsigned rankArg = PA.getWorkGroupRank();
        unsigned numRanks = PA.getWorkGroupSize();
        bool useMPI = PA.isParallel();

        nParticles = size;
        usingMPI = useMPI;
        nBuckets = numBuckets;
        nRanks = numRanks;
        rank = rankArg;

        if( ! usingMPI ) {
            particlesPerRank = nParticles / nRanks;
        } else {
            particlesPerRank = nParticles / nRanks;
        }

        particlesPerRank = std::max( 1U, particlesPerRank);
        particlesPerBucket = std::max( 1U, particlesPerRank / nBuckets);

        ptrBucketHeader = nullptr;
        ptrLocalCollisionPointList = nullptr;
        ptrRankInfo = nullptr;

        unsigned allocationRanks = nRanks-1; // no allocation for rank 0
        unsigned offset = 1; // need for MPI 3 shared memory padding for rank 0

        if( ! usingMPI ) {
            ptrBucketHeader = new bucket_header_t[offset + nBuckets*allocationRanks];
            ptrLocalCollisionPointList = new COLLISION_T[offset + nBuckets*particlesPerBucket*allocationRanks];
            ptrRankInfo = new rank_info_t[nRanks];
        } else if ( usingMPI )  {
            // mpi

            MPI_Info win_info;
            MPI_Info_create(&win_info);
            MPI_Info_set(win_info, "alloc_shared_noncontig", "true");

            if( rank == 0 ) {
                MPI_Win_allocate_shared( (1+nBuckets*allocationRanks)*sizeof(bucket_header_t),
                                         sizeof(bucket_header_t),
                                         win_info,
                                         PA.getWorkGroupCommunicator(),
                                         &ptrBucketHeader,
                                         &bucket_header_shared_memory_window);

                MPI_Win_allocate_shared( (1+nBuckets*particlesPerBucket*allocationRanks)*sizeof(COLLISION_T),
                                         sizeof(COLLISION_T),
                                         win_info,
                                         PA.getWorkGroupCommunicator(),
                                         &ptrLocalCollisionPointList,
                                         &collision_shared_memory_window);

                MPI_Win_allocate_shared( nRanks*sizeof(rank_info_t),
                                         sizeof(rank_info_t),
                                         win_info,
                                         PA.getWorkGroupCommunicator(),
                                         &ptrRankInfo,
                                         &rank_info_shared_memory_window);
            } else {

                MPI_Win_allocate_shared( 0,
                                         sizeof(bucket_header_t),
                                         win_info,
                                         PA.getWorkGroupCommunicator(),
                                         &ptrBucketHeader,
                                         &bucket_header_shared_memory_window);

                MPI_Win_allocate_shared( 0,
                                         sizeof(COLLISION_T),
                                         win_info,
                                         PA.getWorkGroupCommunicator(),
                                         &ptrLocalCollisionPointList,
                                         &collision_shared_memory_window);

                MPI_Win_allocate_shared( 0,
                                         sizeof(rank_info_t),
                                         win_info,
                                         PA.getWorkGroupCommunicator(),
                                         &ptrRankInfo,
                                         &rank_info_shared_memory_window);
            }
            MPI_Info_free(&win_info);

            MPI_Barrier( PA.getWorkGroupCommunicator() );

            // Disabling MPI_Win_sync, causes failure in IBM Spectrum MPI
            // Removal doesn't seem to affect correctness
            //
            // MPI_Win_sync( bucket_header_shared_memory_window );
            // MPI_Win_sync( collision_shared_memory_window );
            // MPI_Win_sync( rank_info_shared_memory_window );
            // MPI_Barrier( PA.getWorkGroupCommunicator() );

            int disp_unit;
            MPI_Aint segment_size;
            MPI_Win_shared_query(bucket_header_shared_memory_window, 0, &segment_size, &disp_unit, &ptrBucketHeader);
            MPI_Win_shared_query(collision_shared_memory_window, 0, &segment_size, &disp_unit, &ptrLocalCollisionPointList);
            MPI_Win_shared_query(rank_info_shared_memory_window, 0, &segment_size, &disp_unit, &ptrRankInfo);

            MPI_Barrier( PA.getWorkGroupCommunicator() );
        }

        rankOffset[RANKINFO].push_back( 0 );
        rankOffset[BUCKETHEADER].push_back( 0 );
        rankOffset[COLLISIONBUFFER].push_back( 0 );
        for(unsigned i = 0; i<allocationRanks; ++i ) {
            rankOffset[RANKINFO].push_back( offset + i );
            rankOffset[BUCKETHEADER].push_back( offset + nBuckets*i );
            rankOffset[COLLISIONBUFFER].push_back( offset + nBuckets*particlesPerBucket*i );
        }

        // initialize headers
        if( rank == 0 ) {
            for( unsigned i = 0; i < nRanks; ++i ){
                ptrRankInfo[i].allDone = false;
                ptrRankInfo[i].currentBucket = 0U;

                if( i > 0 ) {
                    // only have buckets for ranks > 0
                    for(unsigned bucket = 0; bucket<nBuckets; ++bucket ) {
                        bucket_header_t* header = getBucketHeader( i, bucket );
                        header->size = 0U;
                        header->done = false;
                    }
                }
            }
        }
        if ( usingMPI )  {
            MPI_Barrier( PA.getWorkGroupCommunicator() );
        }

        store = [&controller] (const COLLISION_T* pParticle, unsigned N ) {
            controller.add( (const void*) pParticle,N);
        };

        controllerFlush = [&controller] (bool final) { controller.flush(final); };

        controllerClear = [&controller] () {
          controller.clearTally();
        };

        controllerDebugPrint = [&controller] () {
          controller.debugPrint();
        };
    }
public:

    ~SharedRayList() {
        if( ! usingMPI ) {
            if( ptrBucketHeader ) delete [] ptrBucketHeader;
            if( ptrLocalCollisionPointList ) delete [] ptrLocalCollisionPointList;
            if( ptrRankInfo ) delete [] ptrRankInfo;
        } else {
            // explicit free causes failure due to double free.
            //            MPI_Free_mem( ptrBucketHeader );
            //            MPI_Free_mem( ptrLocalCollisionPointList );
            //            MPI_Free_mem( ptrRankInfo );
        }
    }

    store_func_t store;
    controllerFlush_func_t controllerFlush;
    controllerClear_func_t controllerClear;
    controllerClear_func_t controllerDebugPrint;

    void store_single_collision( const COLLISION_T& collision ){
        store(  &collision, 1);
    }

    void store_collision( const COLLISION_T* ptrCollision, unsigned N=1 ){
        store(  ptrCollision, N);
    }

    void master_flush(bool final=false){
        do{
            // keep flushing until all processes done
            flushBuffers();
        } while ( ! allDone() );

        do{
            // keep flushing until all buckets empty
            flushBuffers();
        } while ( ! allEmpty() );
        controllerFlush(final);
    }

    void flushBuffers(){
        if( flushForward ) {
            for( unsigned i = 1; i< nRanks; ++i ) {
                copyToMaster(i);
            }
            flushForward = false;
        } else {
            for( unsigned i = nRanks-1; i > 0; --i ) {
                copyToMaster(i);
            }
            flushForward = true;
        }
        nMaster = 0;
    }

    unsigned size() const { return nParticles;}
    bool isUsingMPI() const { return usingMPI; }
    unsigned getNBuckets() const { return nBuckets; }
    unsigned getNRanks() const { return nRanks; }
    unsigned getParticlesPerRank() const { return particlesPerRank; }
    unsigned getParticlesPerBucket() const { return particlesPerBucket; }

    bucket_header_t* getBucketHeader( unsigned targetRank, unsigned bucketID ) const {
        MONTERAY_ASSERT( targetRank > 0 );
        MONTERAY_ASSERT( targetRank < nRanks );
        MONTERAY_ASSERT( bucketID < nBuckets );

        return &(ptrBucketHeader[ rankOffset[BUCKETHEADER][targetRank] + bucketID ] );
    }

    rank_info_t* getRankInfo( unsigned targetRank ) const {
        MONTERAY_ASSERT( targetRank < nRanks );
        return &(ptrRankInfo[ rankOffset[RANKINFO][targetRank] ]);
    }

    unsigned getCollisionBufferOffset( unsigned targetRank, unsigned bucket ) const {
        MONTERAY_ASSERT( targetRank > 0 );
        return rankOffset[COLLISIONBUFFER][targetRank] + bucket*particlesPerBucket;
    }

    bool isBucketFull( unsigned targetRank, unsigned bucketID ) const {
        if( targetRank == 0 ) return false;
        if( isBucketDone(targetRank, bucketID ) ) {
            return true;
        }

        bucket_header_t* header = getBucketHeader( targetRank, bucketID );
        if( header->size >= particlesPerBucket ) return true;
        return false;
    }

    bool isBucketDone( unsigned targetRank, unsigned bucketID, int offset = 0 ) const {
        if( targetRank == 0 ) return false;

        bucket_header_t* header = getBucketHeader( targetRank, bucketID );
        
        if( header->done.load() ) {
            return true;
        } else {
            return false;
        }
    }

    unsigned bucketSize( unsigned targetRank, unsigned bucketID, int offset = 0 ) const {
        if( targetRank == 0 ) return 0;

        bucket_header_t* header = getBucketHeader( targetRank, bucketID );
        return header->size;
    }

    void addCollisionLocal(unsigned targetRank, const COLLISION_T& collision) {
        MONTERAY_ASSERT( targetRank < nRanks );

#ifndef NDEBUG
        const bool debug = false;
#endif

        // store particle in local memory
        bool stored = false;
        unsigned nPasses = 0;
        unsigned& currentBucket = getRankInfo(targetRank)->currentBucket;
        MonteRay::cpuTimer timer;
        do{

            bucket_header_t* header = getBucketHeader( targetRank, currentBucket );
            if( ! header->done ) {
                bucket_header_t* header = getBucketHeader( targetRank, currentBucket );
                //            if( header->size < particlesPerBucket ) {
                //                if( header->size == 0 ) {
                //                   header->done = false;
                //                }
                // store particle
                ptrLocalCollisionPointList[ getCollisionBufferOffset(targetRank,currentBucket)  + header->size] = collision;
                ++header->size;

#ifndef NDEBUG
                if( debug ) printf( "Debug: addCollisionLocal - add particle, rank = %d, bucket = %d, size = %d\n", targetRank, currentBucket, header->size );
#endif

                stored = true;
                if( header->size == particlesPerBucket ) {
                    header->done.store( true );
                    ++currentBucket;
                }
            } else {
                ++currentBucket;
            }

            if( currentBucket >= nBuckets ){
                currentBucket = 0;
                ++nPasses;

                if( nPasses == 1000 ) {
                    timer.start();
                }

                if( (nPasses > 1000) && (nPasses % 1000 == 0) ) {
                    timer.stop();

                    // 60 sec timeout
                    if( timer.getTime() > 60.0 ) {
                        throw std::runtime_error( "SharedCollision::addCollisionLocal - waiting too long for bucket cleaning, stopping!" );
                    }
                }
            }
        } while( ! stored );
    }

    void addCollisionMaster( const COLLISION_T& collision) {
        store_single_collision( collision );

        // flush the other buffers every time master has generated enough
        // collisions to fill a bucket.
        ++nMaster;
        if( nMaster >= particlesPerBucket ){
            flushBuffers();
        }
    }

    void addCollision( unsigned targetRank, const COLLISION_T& collision) {


        MONTERAY_ASSERT( targetRank < nRanks );

        if( targetRank > 0 ) {
            addCollisionLocal(targetRank, collision);
        } else {
            addCollisionMaster(collision);
        }
    }

    void add( const COLLISION_T& collision) {
        addCollision( rank, collision );
    }

    void addCollision2( unsigned targetRank, float_t pos[3], float_t dir[3], float_t energy, float_t weight, unsigned index) {
        COLLISION_T collision;
        std::memcpy( &collision.pos, pos, 3*sizeof(float_t));
        std::memcpy( &collision.dir, dir, 3*sizeof(float_t));
        collision.energy[0] = energy;
        collision.weight[0] = weight;
        collision.index = index;

        addCollision( targetRank, collision);
    }

    // enable addCollision for triple energy, probability pairs
    template<typename PARTICLE_T, typename SCATTERING_PROBABILITES,
             typename Foo = COLLISION_T,
             typename std::enable_if<(Foo::getN() == 3)>::type* = nullptr >
    void add( const PARTICLE_T& particle,
              const SCATTERING_PROBABILITES& results,
              unsigned detectorIndex) {

        COLLISION_T collision( particle, results, detectorIndex);
        addCollision( rank, collision);
    }

    // enable addCollision for a single energy, probability pair
    template<typename PARTICLE_T,
             typename Foo = COLLISION_T,
             typename std::enable_if<(Foo::getN() == 1)>::type* = nullptr >
    void add( const PARTICLE_T& particle, double probability ) {
        COLLISION_T collision( particle, probability );
        addCollision( rank, collision);
    }

    unsigned getCurrentBucket(unsigned targetRank) const {
        if( targetRank == 0 ) return 0;

        return getRankInfo(targetRank)->currentBucket;
    }

    void copyToMaster( unsigned targetRank ){
        if( rank != 0 ) {
            throw std::runtime_error("SharedRayList::copyToMaster -- can only perform operation from rank 0 ");
        }

        for( unsigned i=0; i<nBuckets; ++i ){
            // search for full or done buckets;
            bucket_header_t* header = getBucketHeader( targetRank, i );
            if( isBucketDone(targetRank, i) && header->size > 0 ) {
                unsigned offset = getCollisionBufferOffset(targetRank,i);
                store_collision( ptrLocalCollisionPointList + offset, header->size  );
                header->size = 0;
                header->done.store( false );
            }
        }
    }

    unsigned getMasterSize() const {
        return nMaster;
    }

    COLLISION_T getCollisionFromLocal( unsigned targetRank, unsigned bucket, unsigned i ){
#ifndef NDEBUG
        const bool debug = false;
#endif

        if( rank != 0 ) {
            throw std::runtime_error("SharedRayList::getCollisionFromLocal -- can only perform operation from rank 0 ");
        }
        unsigned offset = getCollisionBufferOffset(targetRank,bucket);
        unsigned index = offset + i;

#ifndef NDEBUG
        if( debug ) printf( "Debug: getCollisionFromLocal -- index = %d, x=%f\n",index,ptrLocalCollisionPointList[index].pos[0]);
#endif

        return ptrLocalCollisionPointList[ index ];
    }

    bool allDone() const {
        if( rank != 0 ) {
            throw std::runtime_error("SharedRayList::allDone -- can only call from rank 0.");
        }
        for( unsigned i = 1; i< nRanks; ++i ){
            if( isRankDone(i) == false ) {
                return false;
            }
        }
        return true;
    }

    bool isRankDone(unsigned targetRank) const {
        MONTERAY_ASSERT( targetRank < nRanks );

        return getRankInfo(targetRank)->allDone.load();
    }

    bool allEmpty() const {
        if( rank != 0 ) {
            throw std::runtime_error("SharedRayList::allEmpty -- can only call from rank 0.");
        }
        for( unsigned i = 1; i< nRanks; ++i ){

            for( unsigned bucket=0; bucket<nBuckets; ++bucket ){
                // search for full or done buckets;
                bucket_header_t* header = getBucketHeader( i, bucket );
                if( header->size != 0 ) {
                    return false;
                }
            }
        }
        return true;
    }

    void flush( bool final=false ) {
        flushRank( PA.getWorkGroupRank(), final);
    }

    void flushRank( unsigned targetRank, bool final=false) {
        MONTERAY_ASSERT( targetRank < nRanks );

        if( targetRank == 0 ) {
            // rank 0 begins polling other ranks before finishing
            master_flush(final);
        } else {
            // mark all buckets as done
            for( unsigned bucket=0; bucket<nBuckets; ++bucket ){
                // search for full or done buckets;
                bucket_header_t* header = getBucketHeader( targetRank, bucket );
                header->done.store( true );
            }

            // mark process as done;
            getRankInfo(targetRank)->allDone.store( true );
        }
    }

    void clear(bool clearController=true) {
        clearRank( PA.getWorkGroupRank(), clearController );
    }

    void clearRank(unsigned targetRank, bool clearController=true) {
        MONTERAY_ASSERT( targetRank < nRanks );

        if ( usingMPI ) MPI_Barrier( PA.getWorkGroupCommunicator() );
        if( targetRank == 0 ) {
            if( clearController ) {
                controllerClear();
            }
            nMaster = 0;
        } else {
            getRankInfo(targetRank)->allDone.store( false );
            for( unsigned bucket=0; bucket<nBuckets; ++bucket ){
                // search for full or done buckets;
                bucket_header_t* header = getBucketHeader( targetRank, bucket );
                header->size = 0U;
                header->done.store( false );
            }
        }
        if ( usingMPI ) MPI_Barrier( PA.getWorkGroupCommunicator() );
    }

    void restart() {
        unsigned targetRank = PA.getWorkGroupRank();
        MONTERAY_ASSERT( targetRank < nRanks );
        if ( usingMPI ) MPI_Barrier( PA.getWorkGroupCommunicator() );
        if( targetRank == 0 ) {
            nMaster = 0;
        } else {
            getRankInfo(targetRank)->allDone.store( false );
            for( unsigned bucket=0; bucket<nBuckets; ++bucket ){
                // reset buckets;
                bucket_header_t* header = getBucketHeader( targetRank, bucket );
                header->size = 0U;
                header->done.store( false );
            }
        }
        if ( usingMPI ) MPI_Barrier( PA.getWorkGroupCommunicator() );
    }


    void debugPrint() {
        controllerDebugPrint();
    }

private:
    unsigned nParticles = 0;
    bool usingMPI = false ;
    unsigned nBuckets = 1;
    unsigned nRanks = 0;
    unsigned rank = 0;
    unsigned particlesPerRank = 1;
    unsigned particlesPerBucket = 1;
    unsigned nMaster = 0;
    bool flushForward = true;

    std::array<std::vector<int>, 3> rankOffset;

    bucket_header_t*  ptrBucketHeader = nullptr;
    rank_info_t*      ptrRankInfo = nullptr;
    COLLISION_T* ptrLocalCollisionPointList = nullptr;

    MPI_Win bucket_header_shared_memory_window;
    MPI_Win collision_shared_memory_window;
    MPI_Win rank_info_shared_memory_window;

    const MonteRayParallelAssistant& PA;
};

} /* namespace MonteRay*/

#endif /* IPCOMM_SHAREDRAYLIST_HH_ */
