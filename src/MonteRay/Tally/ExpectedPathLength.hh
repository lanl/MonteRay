#ifndef MR_EXPECTEDPATHLENGTH_HH_
#define MR_EXPECTEDPATHLENGTH_HH_

#include <limits>

#include <functional>
#include <memory>

#include "MaterialProperties.hh"
#include "RayListInterface.hh"
#include "RayWorkInfo.hh"
#include "Tally.hh"

namespace MonteRay{

class ExpectedPathLengthTally: public Tally {
  private:
    ExpectedPathLengthTally(Tally tally): Tally(std::move(tally)) {}

  public:
    using Tally::Tally;
    using TallyFloat = Tally::TallyFloat;

  template<typename Ray, typename Geometry, typename MaterialList>
  CUDA_CALLABLE_MEMBER 
  void tallyCollision(
          int particleID,
          const Geometry& geometry,
          const MaterialList& matList,
          const MaterialProperties& matProps,
          const Ray& p,
          RayWorkInfo& rayInfo);

  template <typename Ray, typename MaterialList>
  CUDA_CALLABLE_MEMBER
  TallyFloat tallyCellSegment(const MaterialList& pMatList,
          const MaterialProperties& pMatProps,
          const gpuFloatType_t* const materialXS,
          unsigned cell,
          gpuRayFloat_t distance,
          const Ray& p,
          TallyFloat opticalPathLength);

  template<typename Ray, typename Geometry, typename MaterialList>
  CUDA_CALLABLE_MEMBER 
  void rayTraceOnGridWithMovingMaterials(
          Ray ray,
          gpuRayFloat_t timeRemaining,
          const Geometry& geometry,
          const MaterialProperties& matProps,
          const MaterialList& matList);

  template<unsigned N, typename Geometry, typename MaterialList>
  void rayTraceTallyWithMovingMaterials(
            const RayList_t<N>* const collisionPoints,
            gpuFloatType_t timeRemaining,
            const Geometry* const geometry,
            const MaterialProperties* const matProps,
            const MaterialList* const matList,
            cudaStream_t* stream = nullptr);

  class Builder: public Tally::Builder {
    public:
    Builder() = default;
    auto build() { 
      return ExpectedPathLengthTally(Tally::Builder::build());
    }
  };
};

template<unsigned N, typename Geometry, typename MaterialList>
CUDA_CALLABLE_KERNEL 
rayTraceTally(const Geometry* const pGeometry,
        const RayList_t<N>* const pCP,
        const MaterialList* const pMatList,
        const MaterialProperties* const pMatProps,
        RayWorkInfo* const pRayInfo,
        ExpectedPathLengthTally* const tally);

} /* end namespace */

#endif /* MR_EXPECTEDPATHLENGTH_HH_ */
