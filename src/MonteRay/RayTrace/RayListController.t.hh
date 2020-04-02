#ifndef RAYLISTCONTROLLER_T_HH_
#define RAYLISTCONTROLLER_T_HH_

#include "RayListController.hh"

#include "ReadAndWriteFiles.hh"
#include "ExpectedPathLength.t.hh"
#include "NextEventEstimator.t.hh"

namespace MonteRay {

template<typename Geometry, unsigned N>
RayListController<Geometry,N>::RayListController(
  int nBlocks,
  int nThreads,
  const Geometry* const pGeometry,
  const MaterialList* const pMatList,
  MaterialProperties* const pMatProps,
  TallyPointerVariant pTally,
  std::string outputFileName,
  size_t capacity):
    nBlocks_(nBlocks), 
    nThreads_(nThreads), 
    pGeometry_(pGeometry), 
    pMatList_(pMatList), 
    pMatProps_(pMatProps), 
    pTally_(std::move(pTally)),
    outputFileName_(std::move(outputFileName)),
    PA_(MonteRayParallelAssistant::getInstance()) { 
  if(PA_.get().getWorkGroupRank() != 0) { return; }

  // init banks
  if(PA_.get().getWorkGroupRank() != 0) { return; }
  bank1_ = std::make_unique<RayListInterface<N>>(capacity);
  bank2_ = std::make_unique<RayListInterface<N>>(capacity);
  currentBank_ = bank1_.get();

  // init RayWorkInfo
  auto launchBounds = setLaunchBounds(nThreads_, nBlocks_, capacity);
  unsigned totalNumThreads  = launchBounds.first * launchBounds.second;
  rayInfo_ = std::make_unique<RayWorkInfo>(totalNumThreads);

  pTimer_.reset(new cpuTimer);
#ifdef __CUDACC__
  stream1_.reset(new cudaStream_t);
  startGPU_.reset(new cudaEvent_t);
  stopGPU_.reset(new cudaEvent_t);
  start_.reset(new cudaEvent_t);
  stop_.reset(new cudaEvent_t);
  copySync1_.reset(new cudaEvent_t);
  copySync2_.reset(new cudaEvent_t);

  cudaStreamCreate(stream1_.get());
  cudaEventCreate(start_.get());
  cudaEventCreate(stop_.get());
  cudaEventCreate(startGPU_.get());
  cudaEventCreate(stopGPU_.get());
  cudaEventCreate(copySync1_.get());
  cudaEventCreate(copySync2_.get());
  currentCopySync_ = copySync1_.get();
#endif

} 

template<typename Geometry, unsigned N>
void RayListController<Geometry,N>::kernel(){

  if (isSendingToFile()) {
    return; // do nothing
  } else if (isUsingExpectedPathLengthTally()){
    if(PA_.get().getWorkGroupRank() != 0) { return; }

    auto launchBounds = setLaunchBounds(nThreads_, nBlocks_, currentBank_->getPtrPoints()->size());

#ifndef NDEBUG
    size_t freeMemory = 0;
    size_t totalMemory = 0;
#ifdef __CUDACC__
    cudaError_t memError = cudaMemGetInfo(&freeMemory, &totalMemory);
    freeMemory = freeMemory/1000000;
    totalMemory = totalMemory/1000000;
#endif
    std::cout << "MonteRay::RayListController -- launching kernel on " <<
                 PA_.get().info() << " with " << launchBounds.first << " blocks, " << launchBounds.second  <<
                 " threads, to process " << currentBank_->getPtrPoints()->size() << " rays," <<
                 " free GPU memory= " << freeMemory << "MB, total GPU memory= " << totalMemory << "MB \n";
#endif

    auto& pExpectedPathLengthTally = mpark::get<ExpectedPathLengthTallyPointer>(pTally_);
    if (pMatProps_->usingMaterialMotion()){
      constexpr gpuFloatType_t timeRemaining = 10.0E6;
      pExpectedPathLengthTally->rayTraceTallyWithMovingMaterials(
          currentBank_->getPtrPoints(),
          timeRemaining,
          pGeometry_,
          pMatProps_,
          pMatList_,
          stream1_.get());
    } else {
#ifdef __CUDACC__
      rayTraceTally<<<launchBounds.first,launchBounds.second,0, *stream1_>>>(
              pGeometry_,
              currentBank_->getPtrPoints(),
              pMatList_,
              pMatProps_,
              rayInfo_.get(),
              pExpectedPathLengthTally.get());
#else
      rayTraceTally(
              pGeometry_,
              currentBank_->getPtrPoints(),
              pMatList_,
              pMatProps_,
              rayInfo_.get(),
              pExpectedPathLengthTally.get());
#endif
    }
  } else if (isUsingNextEventEstimator()){
    auto& pNextEventEstimator = mpark::get<NextEventEstimatorPointer>(pTally_);
    if(currentBank_->size() > 0) {
      launch_ScoreRayList(pNextEventEstimator.get(), nBlocks_, nThreads_, currentBank_->getPtrPoints(), rayInfo_.get(), 
          pGeometry_, pMatProps_, pMatList_, stream1_.get());
    }
  }
}

template<typename Geometry, unsigned N>
unsigned
RayListController<Geometry,N>::capacity(void) const {
  return currentBank_ ? currentBank_->capacity() : 0;
}

template<typename Geometry, unsigned N>
unsigned
RayListController<Geometry,N>::size(void) const {
  return currentBank_ ? currentBank_->size() : 0;
}

template<typename Geometry, unsigned N>
unsigned
RayListController<Geometry,N>::getWorldRank() {
    return PA_.get().getWorldRank();
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::flush(bool final){
  if(PA_.get().getWorkGroupRank() != 0) { return; }

  if(isSendingToFile()) { 
    flushToFile(final); }

  if(currentBank_->size() == 0) {
    if(final) {
      deviceSynchronize();
      printTotalTime();
      deviceSynchronize();
    }
    return;
  }

  if(nFlushs_ > 0) {
    stopTimers();
  }

  startTimers();

  ++nFlushs_;

#ifdef __CUDACC__
  gpuErrchk(cudaPeekAtLastError());
  currentBank_->copyToGPU();
  /* gpuErrchk(cudaEventRecord(*currentCopySync_, 0)); */
  /* gpuErrchk(cudaEventSynchronize(*currentCopySync_)); */
  gpuErrchk(defaultStreamSync());
#endif

  // launch kernel
  kernel();

  // only uncomment for testing, forces the cpu and gpu to sync
#ifndef NDEBUG
#ifdef __CUDACC__
  gpuErrchk(cudaPeekAtLastError());
#endif
#endif

#ifdef __CUDACC__
  gpuErrchk(cudaEventRecord(*stopGPU_, *stream1_));
  gpuErrchk(cudaStreamWaitEvent(*stream1_, *stopGPU_, 0));
#endif

  if(final) {
    std::cout << "MonteRay::RayListController: final flush nFlushs_ = " <<nFlushs_-1 << " -- stopping timers\n";
    stopTimers();
    printTotalTime();
    currentBank_->clear();
    return;
  }
  swapBanks();
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::flushToFile(bool final){
  if(PA_.get().getWorldRank() != 0) { return; }

  if(! fileIsOpen_) {
    try {
      currentBank_->openOutput(outputFileName_);
    } catch (...) {
      std::stringstream msg;
      msg << "Failure opening file for collision writing!\n";
      msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "RayListController::flushToFile" << "\n\n";
      std::cout << "MonteRay Error: " << msg.str();
      throw std::runtime_error(msg.str());
    }
    fileIsOpen_ = true;
  }

  try {
    currentBank_->writeBank();
  } catch (...) {
    std::stringstream msg;
    msg << "Failure writing collisions to file!\n";
    msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " << "RayListController::flushToFile" << "\n\n";
    std::cout << "MonteRay Error: " << msg.str();
    throw std::runtime_error(msg.str());
  }
  currentBank_->clear();

  if(final) {
    try {
      currentBank_->closeOutput();
    } catch (...) {
      std::stringstream msg;
      msg << "Failure closing collision file!\n";
      msg << "Called from : " << __FILE__ << "[" << __LINE__ << "] : " <<"RayListController::flushToFile" << "\n\n";
      std::cout << "MonteRay Error: " << msg.str();
      throw std::runtime_error(msg.str());
    }
    fileIsOpen_ = false;
  }
}

template<typename Geometry, unsigned N>
size_t
RayListController<Geometry,N>::readCollisionsFromFile(std::string name) {
  if(PA_.get().getWorldRank() != 0) { return 0; }

  bool end = false;
  unsigned numParticles = 0;

  do  {
    end = currentBank_->readToBank(name, numParticles);
    numParticles += currentBank_->size();
    flush(end);
  } while (! end);
  return numParticles;
}

template<typename Geometry, unsigned N>
size_t
RayListController<Geometry,N>::readCollisionsFromFileToBuffer(std::string name){
  if(PA_.get().getWorldRank() != 0) { return 0; }

  unsigned numParticles = 0;
  currentBank_->readToBank(name, numParticles);
  numParticles += currentBank_->size();
  return numParticles;
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::startTimers(){
  // start timers
  if(PA_.get().getWorkGroupRank() != 0) { return; }

  pTimer_->start();
#ifdef __CUDACC__
  gpuErrchk(cudaEventRecord(*start_,0));
  gpuErrchk(cudaEventRecord(*startGPU_, *stream1_));
#endif
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::stopTimers(){
  // stop timers and sync
  if(PA_.get().getWorkGroupRank() != 0) { return; }

  pTimer_->stop();
  float_t cpuCycleTime = pTimer_->getTime();
  cpuTime_ += cpuCycleTime;

#ifdef __CUDACC__
  gpuErrchk(cudaStreamSynchronize(*stream1_));
  gpuErrchk(cudaEventRecord(*stop_, 0));
  gpuErrchk(cudaEventSynchronize(*stop_));

  float_t gpuCycleTime;
  gpuErrchk(cudaEventElapsedTime(&gpuCycleTime, *startGPU_, *stopGPU_));
  gpuCycleTime /= 1000.0;
  if(gpuCycleTime < 0.0) {
      gpuCycleTime = 0.0;
  }
  gpuTime_ += gpuCycleTime;

  float totalCycleTime;
  gpuErrchk(cudaEventElapsedTime(&totalCycleTime, *start_, *stop_));
  totalCycleTime /= 1000.0;
  wallTime_ += totalCycleTime;
  printCycleTime(cpuCycleTime, gpuCycleTime , totalCycleTime);
#else
  printCycleTime(cpuCycleTime, cpuCycleTime , cpuCycleTime);
#endif

}

template<typename Geometry, unsigned N>
void RayListController<Geometry,N>::swapBanks(){
  if(PA_.get().getWorkGroupRank() != 0) { return; }

  // Swap banks
  if(currentBank_ == bank1_.get()) {
      currentBank_ = bank2_.get();
#ifdef __CUDACC__
      currentCopySync_ = copySync2_.get();
#endif
  } else {
      currentBank_ = bank1_.get();
#ifdef __CUDACC__
      currentCopySync_ = copySync1_.get();
#endif
  }

#ifdef __CUDACC__
  cudaEventSynchronize(*currentCopySync_);
#endif
  currentBank_->clear();
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::sync(void){
  if(PA_.get().getWorkGroupRank() != 0) { return; }
  deviceSynchronize();
}

template<typename Geometry, unsigned N>
void
RayListController<Geometry,N>::clearTally(void) {
  if(PA_.get().getWorkGroupRank() != 0) { return; }

  if(nFlushs_ > 0) {
      stopTimers();
  }
  //	std::cout << "MonteRay::clearTally nFlushs_ = " << nFlushs_ << " -- starting timers\n";
  //	startTimers();
  //
  //	++nFlushs_;

  deviceSynchronize();
  mpark::visit([] (auto& pTally) { pTally->clear(); }, pTally_);
  if(bank1_) bank1_->clear();
  if(bank2_) bank2_->clear();
  deviceSynchronize();
}

template<typename Geometry, unsigned N>
void RayListController<Geometry,N>::printTotalTime() const{
  std::cout << "\n";
  std::cout << "MonteRay::RayListController: total gpuTime = " << gpuTime_ << "\n";
  std::cout << "MonteRay::RayListController: total cpuTime = " << cpuTime_ << "\n";
  std::cout << "MonteRay::RayListController: total wallTime = " << wallTime_ << "\n";
}

template<typename Geometry, unsigned N>
void RayListController<Geometry,N>::printCycleTime(float_t cpu, float_t gpu, float_t wall) const{
  std::cout << "\n";
  std::cout << "MonteRay::RayListController: cycle gpuTime = " << gpu << "\n";
  std::cout << "MonteRay::RayListController: cycle cpuTime = " << cpu << "\n";
  std::cout << "MonteRay::RayListController: cycle wallTime = " << wall << "\n";
}

template<typename Geometry, unsigned N>
void RayListController<Geometry,N>::printPointDets(const std::string& outputFile, unsigned nSamples, unsigned constantDimension) {
  if(PA_.get().getWorldRank() != 0) { return; }
  if (isUsingNextEventEstimator()){
    auto& pNextEventEstimator = mpark::get<NextEventEstimatorPointer>(pTally_);
    pNextEventEstimator->printPointDets(outputFile, nSamples, constantDimension);
  } else {
    throw std::runtime_error("MonteRay::RayListController::printPointDets - controller does not control a NextEventEstimator.");
  }
}

template<typename Geometry, unsigned N>
void RayListController<Geometry,N>::outputTimeBinnedTotal(std::ostream& outputFile, unsigned nSamples, unsigned constantDimension){
  if(PA_.get().getWorldRank() != 0) { return; }
  if (isUsingNextEventEstimator()){
    auto& pNextEventEstimator = mpark::get<NextEventEstimatorPointer>(pTally_);
    pNextEventEstimator->outputTimeBinnedTotal(outputFile, nSamples, constantDimension);
  } else {
    throw std::runtime_error("MonteRay::RayListController::outputTimeBinnedTotal - controller does not control a NextEventEstimator.");
  }
}

template<typename Geometry, unsigned N>
void RayListController<Geometry,N>::updateMaterialProperties(MaterialProperties* pMPs) {
  if(PA_.get().getWorkGroupRank() != 0) { return; }
  pMatProps_ = pMPs;
}

template<typename Geometry, unsigned N>
void RayListController<Geometry,N>::copyPointDetTallyToCPU(void) {
  if(PA_.get().getWorkGroupRank() != 0) { return; }
  if(! isUsingNextEventEstimator()) {
    throw std::runtime_error("RayListController::copyPointDetTallyToCPU - Next-Event Estimator not enabled.");
  }
#ifdef __CUDACC__
  deviceSynchronize();
#endif
}

template<typename Geometry, unsigned N>
void RayListController<Geometry,N>::writeTalliesToFile(const std::string& fileName) {
  std::ofstream out(fileName, std::ios::out);
  if (isUsingNextEventEstimator()){
    binaryIO::write(out, "NEE");
  } else if (isUsingExpectedPathLengthTally()) {
    binaryIO::write(out, "EPL");
  } else {
    binaryIO::write(out, "None");
  }
  mpark::visit([&] (auto& pTally) { pTally->write(out); }, pTally_);
}

template<typename Geometry, unsigned N>
void RayListController<Geometry,N>::gather() {
  if(PA_.get().getWorkGroupRank() != 0) { return; }
  mpark::visit([] (auto& pTally) { pTally->gather(); }, pTally_);
}

template<typename Geometry, unsigned N>
void RayListController<Geometry,N>::accumulate() {
  if(PA_.get().getWorldRank() != 0) { return; }
  mpark::visit([] (auto& pTally) { pTally->accumulate(); }, pTally_);
}

template<typename Geometry, unsigned N>
void RayListController<Geometry,N>::computeStats() {
  if(PA_.get().getWorldRank() != 0) { return; }
  mpark::visit([] (auto& pTally) { pTally->computeStats(); }, pTally_);
}

} // end namespace MonteRay

#endif // RAYLISTCONTROLLER_T_HH
