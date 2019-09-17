#ifndef MONTERAYSINGLEVALUECOPYMEMORY_T_HH_
#define MONTERAYSINGLEVALUECOPYMEMORY_T_HH_

#include "MonteRay_SingleValueCopyMemory.hh"
#include "MonteRayCopyMemory.t.hh"
#include "MonteRayParallelAssistant.hh"

namespace MonteRay {

template<typename T>
MonteRay_SingleValueCopyMemory<T>::MonteRay_SingleValueCopyMemory() :
    CopyMemoryBase<MonteRay_SingleValueCopyMemory<T>>()
{
    init();
}

template<typename T>
void
MonteRay_SingleValueCopyMemory<T>::copyToGPU(void) {
    //if( debug ) std::cout << "Debug: MonteRay_SingleValueCopyMemory::copyToGPU \n";
    if( ! MonteRay::isWorkGroupMaster() ) return;
    Base::copyToGPU();
}

} /* end namespace */

#endif /* MONTERAYSINGLEVALUECOPYMEMORY_T_HH_ */
