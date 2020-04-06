#ifndef RAYLISTCONTROLLER_HH_
#define RAYLISTCONTROLLER_HH_

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <iostream>
#include <algorithm>
#include <mpark/variant.hpp>

#include "MaterialList.hh"
#include "MaterialProperties.hh"
#include "RayListInterface.hh"
#include "ExpectedPathLength.hh"
#include "GPUErrorCheck.hh"
#include "GPUUtilityFunctions.hh"
#include "MonteRay_timer.hh"
#include "MonteRayTypes.hh"
#include "MonteRayParallelAssistant.hh"
#include "RayWorkInfo.hh"
#include "NextEventEstimator.hh"
#include "CudaWrappers.hh"

namespace MonteRay {

class cpuTimer;

template< unsigned N >
class RayListInterface;

template< unsigned N >
class Ray_t;

template<typename Geometry, unsigned N = 1>
class RayListController {
private:
  using ExpectedPathLengthTallyPointer = std::unique_ptr<ExpectedPathLengthTally>;
  using NextEventEstimatorPointer = std::unique_ptr<NextEventEstimator>;
  using TallyPointerVariant = mpark::variant<NextEventEstimatorPointer, ExpectedPathLengthTallyPointer>;
private:
  unsigned nBlocks_ = 0;
  unsigned nThreads_ = 0;
  const Geometry* pGeometry_ = nullptr;
  const MaterialList* pMatList_ = nullptr;
  MaterialProperties* pMatProps_ = nullptr;
  TallyPointerVariant pTally_;
  std::reference_wrapper<const MonteRayParallelAssistant> PA_;

  RayListInterface<N>* currentBank_ = nullptr;
  std::unique_ptr<RayListInterface<N>> bank1_;
  std::unique_ptr<RayListInterface<N>> bank2_;

  std::unique_ptr<RayWorkInfo> rayInfo_;

  unsigned nFlushs_ = 0;

  std::unique_ptr<cpuTimer> pTimer_;
  double cpuTime_ = 0.0;
  double gpuTime_ = 0.0;
  double wallTime_= 0.0;
  bool fileIsOpen_ = false;
  std::string outputFileName_;

  cuda::StreamPointer stream1_;
  cuda::StreamPointer stream2_;
  cuda::EventPointer startGPU_;
  cuda::EventPointer stopGPU_;
  cuda::EventPointer start_;
  cuda::EventPointer stop_;
  cuda::EventPointer copySync1_;
  cuda::EventPointer copySync2_;
  std::reference_wrapper<cuda::EventPointer> currentCopySync_;

  RayListController(
          int nBlocks,
          int nThreads,
          const Geometry* const geometry,
          const MaterialList* const matList,
          MaterialProperties* const matProps,
          TallyPointerVariant pTally,
          std::string outputFileName,
          size_t capacity);

  void stopTimers(void);
  void startTimers(void);
  void printCycleTime(float_t cpu, float_t gpu, float_t wall) const;
  void swapBanks(void);

  void kernel();

public:
  using Ray = Ray_t<N>;

  unsigned capacity(void) const;
  unsigned size(void) const;

  void addRay(const Ray& ray){
    if(MonteRayParallelAssistant::getInstance().getWorkGroupRank() != 0) { return; }
    currentBank_->add(ray);
    if(size() == capacity()) {
      flush();
    }
  }

  void add(const Ray& ray){ addRay(ray); }
  template <typename Rays>
  void addRays(Rays&& rays){
    for (const auto& ray: rays){
      addRay(ray);
    }
  }
  void add(const Ray* const rayArray, unsigned num=1){
    addRays(SimpleView<const Ray>(rayArray, rayArray + num));
  }
  void add(const void* const rayPtr, unsigned num = 1){ add(static_cast<const Ray*>(rayPtr), num); }


  unsigned addPointDet(gpuFloatType_t x, gpuFloatType_t y, gpuFloatType_t z, gpuFloatType_t exclusionRad = 0);
  void setPointDetExclusionRadius(gpuFloatType_t r);
  void copyPointDetTallyToCPU(void);
  void copyPointDetToGPU(void);
  void printPointDets(const std::string& outputFile, unsigned nSamples, unsigned constantDimension=2);
  void outputTimeBinnedTotal(std::ostream& out,unsigned nSamples=1, unsigned constantDimension=2);
  CUDAHOST_CALLABLE_MEMBER void updateMaterialProperties(MaterialProperties* pMPs);
  void writeTalliesToFile(const std::string& fileName);

  void flush(bool final=false);

public:
  void printTotalTime(void) const;

  double getCPUTime(void) const { return cpuTime_; }
  double getGPUTime(void) const { return gpuTime_; }
  unsigned getNFlushes(void) const { return nFlushs_; }

  void sync(void);

  void clearTally(void);

  size_t readCollisionsFromFile(std::string name);

  // reads a single block of rays to a buffer but doesn't flush them
  // usually for testing or debugging
  size_t readCollisionsFromFileToBuffer(std::string name);

  void flushToFile(bool final=false);

  void debugPrint() { currentBank_->debugPrint(); }

  bool isSendingToFile(void) { return outputFileName_.size() > 0; }
  bool isUsingNextEventEstimator(void) const { 
    return mpark::holds_alternative<NextEventEstimatorPointer>(pTally_);
  }
  bool isUsingExpectedPathLengthTally(void) const { 
    return mpark::holds_alternative<ExpectedPathLengthTallyPointer>(pTally_);
  }

  unsigned getWorldRank();

  void gather();
  void accumulate();
  void computeStats();

  template <typename... Args>
  auto contribution(Args... args){
    return mpark::visit([args...] (auto& pTally) { return pTally->contribution(args...); }, pTally_);
  }
  template <typename... Args>
  auto mean(Args... args){
    return mpark::visit([args...] (auto& pTally) { return pTally->mean(args...); }, pTally_);
  }
  template <typename... Args>
  auto stdDev(Args... args){
    return mpark::visit([args...] (auto& pTally) { return pTally->stdDev(args...); }, pTally_);
  }

  class Builder {
    private:
    TallyPointerVariant b_pTally_;
    unsigned b_nBlocks_ = 18;
    unsigned b_nThreads_ = 256;
    const Geometry* b_pGeometry_ = nullptr;
    const MaterialList* b_pMatList_ = nullptr;
    MaterialProperties* b_pMatProps_ = nullptr;
    std::string b_outputFileName_;
    size_t b_capacity_ = 100000;

    public:
    auto& nBlocks(unsigned val){ b_nBlocks_ = val; return *this; }
    auto& nThreads(unsigned val){ b_nThreads_ = val; return *this; }
    auto& geometry(const Geometry* const val){ b_pGeometry_ = val; return *this; }
    auto& materialList(const MaterialList* const val){ b_pMatList_= val; return *this; }
    auto& materialProperties(MaterialProperties* const val){ b_pMatProps_ = val; return *this; }
    auto& capacity(size_t val) {b_capacity_ = val; return *this; }
    auto& outputFileName(std::string name) {
      std::cout << "MonteRay::RayListController::Builder::outputFileName - MonteRay will print rays to file " + name + " instead of tallying them." << std::endl;
      b_outputFileName_ = name;
      return *this;
    }

    bool tallyIsSet() const { 
      return mpark::visit( [] (const auto& pTally) { return pTally ? true : false; }, b_pTally_);
    }

    auto& expectedPathLengthTally(ExpectedPathLengthTally tally){
      if (tallyIsSet()){ 
        throw std::runtime_error("MonteRay::RayListController::Builder - tally has already been assigned.");
      }
      b_pTally_ = std::make_unique<ExpectedPathLengthTally>(std::move(tally));
      return *this;
    }

    auto& nextEventEstimator(NextEventEstimator nee){
      if (tallyIsSet()){ 
        throw std::runtime_error("MonteRay::RayListController::Builder - tally has already been assigned.");
      }
      b_pTally_ = std::make_unique<NextEventEstimator>(std::move(nee));
      return *this;
    }
    

    auto build(){
      if ( not tallyIsSet() && b_outputFileName_.size() == 0){ 
        throw std::runtime_error("MonteRay::RayListController::Builder requires at least one ExpectedPathLengthTally or NextEventEstimator or output file name");
      }
      return RayListController(b_nBlocks_, b_nThreads_, b_pGeometry_, b_pMatList_, b_pMatProps_, 
          std::move(b_pTally_), std::move(b_outputFileName_), b_capacity_);
    }
  };
};

}

#include "Geometry.hh"
namespace MonteRay{

using CollisionPointController = typename MonteRay::RayListController<MonteRay_SpatialGrid,1>;
using NextEventEstimatorController = typename MonteRay::RayListController<MonteRay_SpatialGrid,3>;

} /* namespace MonteRay */

#endif /* RAYLISTCONTROLLER_HH_ */
