#include "Tally.hh"
#include "GPUAtomicAdd.hh"
namespace MonteRay{

CUDA_CALLABLE_MEMBER
void Tally::scoreByIndex(gpuTallyType_t value, int spatial_index, int time_index){
  int index = getIndex( spatial_index, time_index );
  gpu_atomicAdd( &( data_[index]), value);
}

}
