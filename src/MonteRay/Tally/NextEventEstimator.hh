#ifndef MONTERAYNEXTEVENTESTIMATOR_HH_
#define MONTERAYNEXTEVENTESTIMATOR_HH_

#define MR_NEE_VERSION static_cast<unsigned>(2)

#include "MonteRayTypes.hh"
#include "MonteRayAssert.hh"
#include "MonteRayVector3D.hh"
#include "ManagedAllocator.hh"
#include "Tally.hh"
#include "RayWorkInfo.hh"
#include "CrossSection.hh"

namespace MonteRay {

template< unsigned N>
class RayList_t;
template< unsigned N>
class Ray_t;

class NextEventEstimator: public Tally {
public:
  using position_t = gpuRayFloat_t;
  using DetectorIndex_t = int;

private:
  Vector<Vector3D<position_t>> tallyPoints_;
  Vector<DataFloat> exclusionRadii_;

  NextEventEstimator(Vector<Vector3D<position_t>> tallyPoints, Vector<DataFloat> exclusionRadii, Tally tally):
    Tally(std::move(tally)),
    tallyPoints_(std::move(tallyPoints)), 
    exclusionRadii_(std::move(exclusionRadii))
  { }
public:

  CUDA_CALLABLE_MEMBER int size(void) const { return tallyPoints_.size(); }

  CUDA_CALLABLE_MEMBER position_t getExclusionRadius(int i) const { return exclusionRadii_[i]; }

  template<unsigned N, typename Geometry, typename MaterialProperties, typename MaterialList>
  CUDA_CALLABLE_MEMBER 
  void calcScore(const int threadID, const Ray_t<N>& ray, RayWorkInfo& rayInfo, 
      const Geometry& geometry, const MaterialProperties& matProps, const MaterialList& matList);

  const auto& getPoint(int i) const { 
    MONTERAY_ASSERT(i<tallyPoints_.size());  
    return tallyPoints_[i]; 
  }

  void printPointDets(const std::string& outputFile, int nSamples, int constantDimension=2) const;
  void outputTimeBinnedTotal(std::ostream& out, int nSamples=1, int constantDimension=2) const;

  template<typename Stream>
  void write(Stream& out){
    binaryIO::write(out, MR_NEE_VERSION);
    binaryIO::write(out, tallyPoints_);
    binaryIO::write(out, exclusionRadii_);
    Tally::write(out);
  }

  class Builder : public Tally::Builder {
    private:
    Vector<DataFloat> b_exclusionRadii_;
    Vector<Vector3D<position_t>> b_tallyPoints_;

    public:
    Builder(int num = 1){
      b_tallyPoints_.reserve(num);
      b_exclusionRadii_.reserve(num);
    }

    auto& addTallyPoint(position_t x, position_t y, position_t z, position_t exclusionRad = 0){
     b_tallyPoints_.emplace_back(x, y, z);
     b_exclusionRadii_.emplace_back(exclusionRad);
     return *this;
    }

    auto build() {
      this->spatialBins(b_tallyPoints_.size());
      auto tally = Tally::Builder::build();
      return NextEventEstimator(
          std::move(b_tallyPoints_), 
          std::move(b_exclusionRadii_),
          std::move(tally));
    }

    template<typename Stream>
    static auto read(Stream& in){
      unsigned version;
      binaryIO::read(in, version);
      if (version != MR_NEE_VERSION){
        throw std::runtime_error("NextEventEstimator dump file version number " + 
            std::to_string(version) + " is incompatible with expected version " + 
            std::to_string(MR_NEE_VERSION));
      }

      Vector<Vector3D<position_t>> tallyPoints;
      binaryIO::read(in, tallyPoints);

      Vector<DataFloat> exclusionRadii;
      binaryIO::read(in, exclusionRadii);

      auto tally = Tally::Builder::read(in);

      return NextEventEstimator(
          std::move(tallyPoints), 
          std::move(exclusionRadii),
          std::move(tally));
    }
  };
};

template<unsigned N, typename Geometry, typename MaterialProperties, typename MaterialList>
inline void cpuScoreRayList(NextEventEstimator* const pNextEventEstimator, const RayList_t<N>* pRayList, 
    RayWorkInfo* pRayInfo, const Geometry* const pGeometry, const MaterialProperties* const pMatProps, 
    const MaterialList* const pMatList){
  for(auto particleID = 0; particleID < pRayList->size(); particleID++) {
    constexpr int threadID = 0;
    pRayInfo->clear(threadID);
    auto& ray = pRayList->points[particleID];
    pNextEventEstimator->calcScore(threadID, ray, *pRayInfo, *pGeometry, *pMatProps, *pMatList);
  }
}

template<unsigned N, typename Geometry, typename MaterialProperties, typename MaterialList>
CUDA_CALLABLE_KERNEL  kernel_ScoreRayList(NextEventEstimator* ptr, const RayList_t<N>* pRayList, 
      const Geometry* const pGeometry, const MaterialProperties* const pMatProps, const MaterialList* const pMatList);

} /* namespace MonteRay */

#undef MR_NEE_VERSION
#endif /* MONTERAYNEXTEVENTESTIMATOR_HH_ */
