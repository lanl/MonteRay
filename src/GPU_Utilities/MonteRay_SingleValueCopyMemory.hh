/*
 * MonteRay_SingleValueCopyMemory.hh
 *
 *  Created on: Feb 12, 2018
 *      Author: jsweezy
 */

#ifndef MONTERAY_SINGLEVALUECOPYMEMORY_HH_
#define MONTERAY_SINGLEVALUECOPYMEMORY_HH_

#include "MonteRayCopyMemory.hh"

namespace MonteRay  {

template<typename T>
class MonteRay_SingleValueCopyMemory :  public CopyMemoryBase<MonteRay_SingleValueCopyMemory<T>> {
public:
    using Base = CopyMemoryBase<MonteRay_SingleValueCopyMemory<T>>;

    MonteRay_SingleValueCopyMemory();

    ~MonteRay_SingleValueCopyMemory(){}

    std::string className(){ return std::string("resultClass");}

    void init() {
        v = T(0);
    }

    void copyToGPU(void) {
        //if( debug ) std::cout << "Debug: MonteRay_SingleValueCopyMemory::copyToGPU \n";
        Base::copyToGPU();
    }

    void copyToCPU(void) {
        //if( debug ) std::cout << "Debug: MonteRay_SingleValueCopyMemory::copyToCPU \n";
        Base::copyToCPU();
    }

    void copy(const MonteRay_SingleValueCopyMemory<T>* rhs) {
        if( this->debug ) {
            std::cout << "Debug: 1- MonteRay_SingleValueCopyMemory::copy(const resultClass* rhs) \n";
        }

        if( this->isCudaIntermediate && rhs->isCudaIntermediate ) {
            throw std::runtime_error("MonteRay_SingleValueCopyMemory::copy -- can NOT copy CUDA intermediate to CUDA intermediate.");
        }

        if( !this->isCudaIntermediate && !rhs->isCudaIntermediate ) {
            throw std::runtime_error("MonteRay_SingleValueCopyMemory::copy -- can NOT copy CUDA non-intermediate to CUDA non-intermediate.");
        }

        v = rhs->v;
    }

    T v;

};

} // end namespace


#endif /* MONTERAY_SINGLEVALUECOPYMEMORY_HH_ */
