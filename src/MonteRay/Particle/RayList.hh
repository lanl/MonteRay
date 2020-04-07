#ifndef RAYLIST_HH_
#define RAYLIST_HH_

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <iostream>

#include "Ray.hh"
#include "ThirdParty/ManagedAllocator.hh"
#include "SimpleVector.hh"

namespace MonteRay{

typedef unsigned RayListSize_t;

// Note: this could effectively be replaced by a MonteRay::Vector
template<unsigned N = 1>
class RayList_t : public Managed {
public:
    using RAY_T = Ray_t<N>;

    SimpleVector<RAY_T> points;

    /// Takes the size of the list as an argument.
    CUDAHOST_CALLABLE_MEMBER RayList_t(RayListSize_t num = 1){ points.reserve(num); }

    CUDAHOST_CALLABLE_MEMBER std::string className() { return std::string("RayList_t");}

    CUDAHOST_CALLABLE_MEMBER void reallocate(size_t n) { points.reserve(n); }

    CUDA_CALLABLE_MEMBER RayListSize_t size(void) const {
        return points.size();
    }

    CUDA_CALLABLE_MEMBER RayListSize_t capacity(void) const { return points.capacity(); }
    void clear(void) { points.clear(); }

    CUDA_CALLABLE_MEMBER const auto& getPosition(RayListSize_t i) const {
        return points[i].pos;
    }

    CUDA_CALLABLE_MEMBER const auto& getDirection(RayListSize_t i) const {
        return points[i].dir;
    }

    CUDA_CALLABLE_MEMBER gpuFloatType_t getEnergy(RayListSize_t i, unsigned index = 0) const {
        return points[i].energy[index];
    }

    CUDA_CALLABLE_MEMBER gpuFloatType_t getWeight(RayListSize_t i, unsigned index = 0) const {
        return points[i].weight[index];
    }

    CUDA_CALLABLE_MEMBER gpuFloatType_t getTime(RayListSize_t i) const {
        return points[i].time;
    }

    CUDA_CALLABLE_MEMBER unsigned getIndex(RayListSize_t i) const {
        return points[i].index;
    }

    CUDA_CALLABLE_MEMBER DetectorIndex_t getDetectorIndex(RayListSize_t i) const {
        return points[i].detectorIndex;
    }

    CUDA_CALLABLE_MEMBER ParticleType_t getParticleType(RayListSize_t i) const {
        return points[i].particleType;
    }

    RAY_T pop(void) {
#ifndef NDEBUG
        if( points.size() == 0 ) {
            printf("RayList::pop -- no points.  %s %d\n", __FILE__, __LINE__);
            ABORT( "RayList.hh -- RayList::pop" );
        }
#endif

        auto ray = points.back();
        points.pop_back();
        return ray;
    }

    CUDA_CALLABLE_MEMBER RAY_T getParticle(RayListSize_t i) const {
#ifndef NDEBUG
        if( i >= points.size() ) {
            printf("RayList::getParticle -- index exceeds size.  %s %d\n", __FILE__, __LINE__);
            ABORT( "RayList.hh -- RayList::getParticle" );
        }
#endif
        return points[i];
    }

    void add(const RAY_T& point ) {
#ifndef NDEBUG
        if( size() >= capacity() ) {
            printf("RayList::add -- index > number of allocated points.  %s %d\n", __FILE__, __LINE__);
            ABORT( "RayList.hh -- RayList::add" );
        }
#endif
        points.push_back(point);
    }

    CUDA_CALLABLE_MEMBER static unsigned getN(void ) { return N; }

    auto data() { return points.data(); }
    const auto data() const { return points.data(); }
    auto begin() { return points.begin(); }
    const auto begin() const { return points.begin(); }
    auto end() { return points.end(); }
    const auto end() const { return points.end(); }

    void writeToFile( const std::string& filename) const;
    void readFromFile( const std::string& filename);

    template<typename IOTYPE>
    void write(IOTYPE& out) const {
      unsigned version = 0;
      binaryIO::write( out, version );
      binaryIO::write( out, static_cast<RayListSize_t>(points.capacity()) );
      binaryIO::write( out, static_cast<RayListSize_t>(points.size()) );
      for (const auto& point : points){
        point.write(out);
      }
    }

    template<typename IOTYPE>
    void read(IOTYPE& in) {
      unsigned version;
      binaryIO::read( in, version );

      RayListSize_t nAllocated;
      binaryIO::read( in, nAllocated );
      points.reserve( nAllocated );

      RayListSize_t nUsed;
      binaryIO::read( in, nUsed );
      points.resize(nUsed);

      for (auto& point : points){
        point.read(in);
      }
    }

};

typedef RayList_t<1> CollisionPoints;
typedef RayList_t<1> ParticleRayList;
typedef RayList_t<3> PointDetRayList;

} // end namespace



#endif /* RAYLIST_HH_ */
