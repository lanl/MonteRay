#ifndef MONTERAY_TALLY_HH_
#define MONTERAY_TALLY_HH_

#include <vector>
#include <cstring>
#include <string>
#include <memory>

#define MR_TALLY_VERSION static_cast<unsigned>(2)

#include "MonteRayConstants.hh"
#include "MonteRayDefinitions.hh"
#include "MonteRayParallelAssistant.hh"
#include "MonteRayAssert.hh"
#include "Containers.hh"
#include "MonteRayTypes.hh"
#include "GPUAtomicAdd.hh"
#include "Filter.hh"

// TPB TODO: convert everything to int or size_t - probably size_t

namespace MonteRay {

class Tally : public Managed {
public:
  using DataFloat = gpuFloatType_t;
  using TallyFloat = gpuTallyType_t;
  using Filter = BinEdgeFilter<Vector<DataFloat>>;

  struct MeanAndStdDev : public std::tuple<TallyFloat, TallyFloat> {
    using std::tuple<TallyFloat, TallyFloat>::tuple;
    MeanAndStdDev() : std::tuple<TallyFloat, TallyFloat>(0.0, 0.0) {}
    auto sum() const { return std::get<0>(*this); }
    auto sumSq() const { return std::get<1>(*this); }

    auto average() const { return std::get<0>(*this); }
    auto mean() const { return average(); }
    auto stdDev() const { return std::get<1>(*this); }
    auto variance() const { return std::get<1>(*this)*std::get<1>(*this); }
  };

  Tally() = delete;

private:
  int nSpatialBins_;
  Filter energyFilter_;
  Filter timeFilter_;
  Vector<TallyFloat> contributions_;
  Vector<MeanAndStdDev> stats_;
  size_t nSamples_;

  Tally(int nSpatialBins, Filter energyFilter, Filter timeFilter, 
      Vector<TallyFloat> contributions, Vector<MeanAndStdDev> stats, size_t nSamples):
        nSpatialBins_(nSpatialBins), energyFilter_(std::move(energyFilter)), timeFilter_(std::move(timeFilter)),
        contributions_(std::move(contributions)), stats_(std::move(stats)), nSamples_(nSamples) { }

public:

  void clear();
  void accumulate();
  void computeStats();
  bool useStats() const { return stats_.size() > 0; }
  auto nSamples() const { return nSamples_; }

  CUDA_CALLABLE_MEMBER size_t nSpatialBins() const {
      return nSpatialBins_;
  }

  const auto& timeBinEdges() const { return timeFilter_.binEdges(); }
  const auto& energyBinEdges() const { return energyFilter_.binEdges(); }
  CUDA_CALLABLE_MEMBER auto nTimeBins() const { return timeFilter_.size(); }
  CUDA_CALLABLE_MEMBER auto nEnergyBins() const { return energyFilter_.size(); }

  CUDA_CALLABLE_MEMBER
  int getIndex(int spatialIndex, int energyIndex = 0, int timeIndex = 0) const {
    // layout is first by spatial index, then by energy, then by time index

    MONTERAY_ASSERT(spatialIndex < nSpatialBins_);
    MONTERAY_ASSERT(energyIndex < nEnergyBins());
    MONTERAY_ASSERT(timeIndex < nTimeBins());
    return  timeIndex*(nSpatialBins_ * nEnergyBins())  + energyIndex*nSpatialBins_ + spatialIndex;
  }

  CUDA_CALLABLE_MEMBER
  void score(TallyFloat value, int spatialIndex, DataFloat energy = 0.0, DataFloat time = 0.0){
    int index = getIndex(spatialIndex, energyFilter_(energy), timeFilter_(time));
    gpu_atomicAdd(&(contributions_[index]), value);
  }

  TallyFloat contribution(int index){
    MONTERAY_ASSERT(contributions_.size() > index);
    return contributions_[index];
  }

  CUDA_CALLABLE_MEMBER
  TallyFloat contribution(int spatialIndex, int energyIndex, int timeIndex = 0) const {
    int index = getIndex(spatialIndex, energyIndex, timeIndex);
    MONTERAY_ASSERT(contributions_.size() > index);
    return contributions_[index];
  }

  auto mean(size_t i) const { return this->useStats() ? stats_[i].mean() : contributions_[i];  }

  auto stdDev(size_t i) const { 
    if (this->useStats()) {
      return this->useStats() ? stats_[i].stdDev() : contributions_[i];  
    } else {
      return 0.0;
    }
  }

  const auto& stats() const { return stats_; }

  CUDA_CALLABLE_MEMBER
  auto size() const { return contributions_.size(); }

  void gather();
  void gatherWorkGroup();

  template<typename Stream>
  void write(Stream& out){
    binaryIO::write(out, MR_TALLY_VERSION);
    binaryIO::write(out, nSpatialBins_);
    energyFilter_.write(out);
    timeFilter_.write(out);
    binaryIO::write(out, contributions_);
    binaryIO::write(out, stats_);
    binaryIO::write(out, nSamples_);
  }

  class Builder {
    protected:
    int b_nSpatialBins_ = 1;
    Filter b_energyFilter_;
    Filter b_timeFilter_;
    bool b_useStats_ = false;
    public:
    Builder() = default;
    template <typename Container>
    void energyBinEdges(Container&& edges){ b_energyFilter_ = Filter{std::move(edges)}; }
    template <typename Container>
    void timeBinEdges(Container&& edges){ b_timeFilter_ = Filter{std::move(edges)}; }
    void spatialBins(size_t val) { b_nSpatialBins_ = val; }
    void useStats(bool val) { b_useStats_ = val; }

    auto build() {
      if (b_nSpatialBins_ <= 0) {
        throw std::runtime_error("Attempting to set number of spatial bins in a tally to invalid value: " + std::to_string(b_nSpatialBins_));
      }
      auto contributionsSize = b_nSpatialBins_*b_timeFilter_.size()*b_energyFilter_.size();
      Vector<TallyFloat> contributions(contributionsSize, 0.0);
      Vector<MeanAndStdDev> stats;
      if (b_useStats_){ stats.resize(2*contributionsSize); }
      size_t nSamples = 0;
      return Tally(b_nSpatialBins_, std::move(b_energyFilter_), std::move(b_timeFilter_), std::move(contributions), std::move(stats), nSamples);
    }

    template <typename Stream>
    static auto read(Stream& in){
      unsigned version;
      binaryIO::read(in, version);
      if (version != MR_TALLY_VERSION){
        throw std::runtime_error("Tally dump file version number " + 
            std::to_string(version) + " is incompatible with expected version " + 
            std::to_string(MR_TALLY_VERSION));
      }

      int nSpatialBins;
      binaryIO::read(in, nSpatialBins);
      auto energyFilter = Filter::read(in);
      auto timeFilter = Filter::read(in);

      Vector<TallyFloat> contributions;
      binaryIO::read(in, contributions);

      Vector<MeanAndStdDev> stats;
      binaryIO::read(in, stats);

      size_t nSamples;
      binaryIO::read(in, nSamples);

      return Tally(nSpatialBins, std::move(energyFilter), std::move(timeFilter), std::move(contributions), std::move(stats), nSamples);
    }

  };

private:
  void gatherImpl();
};

} // end namespace

#undef MR_TALLY_VERSION 
#endif /* MONTERAYTALLY_HH_ */
