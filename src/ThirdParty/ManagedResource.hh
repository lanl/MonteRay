#ifndef MANAGEDRESOURCE_HH_
#define MANAGEDRESOURCE_HH_

#include <cstdlib>

namespace MonteRay{

class ManagedResource{

  public:
    void* allocate(size_t n);

    void deallocate(void* ptr, size_t);

    bool is_equal(const ManagedResource&){
      return true;
    }

};

}

#endif
