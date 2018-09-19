#ifndef MONTERAYSINGLEVALUECOPYMEMORY_T_HH_
#define MONTERAYSINGLEVALUECOPYMEMORY_T_HH_

#include "MonteRay_SingleValueCopyMemory.hh"
#include "MonteRayCopyMemory.t.hh"

namespace MonteRay {

template<typename T>
MonteRay_SingleValueCopyMemory<T>::MonteRay_SingleValueCopyMemory() :
    CopyMemoryBase<MonteRay_SingleValueCopyMemory<T>>()
{
    init();
}

} /* end namespace */

#endif /* MONTERAYSINGLEVALUECOPYMEMORY_T_HH_ */
