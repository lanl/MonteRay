#ifndef CROSSSECTIONLIST_HH_
#define CROSSSECTIONLIST_HH_

#include "CrossSection.hh"

namespace MonteRay {

class CrossSectionList : public Managed {
public:
    CrossSectionList(){};

    ~CrossSectionList(){};

    void add( CrossSection xs ) {
        CrossSection* ptr = getXSByZAID( xs.ZAID() );
        if( ptr ) { return; }

        xs.setID( list_vec.size() );
        list_vec.push_back(xs);

        list = list_vec.data();
        list_size = list_vec.size();
    }

    CUDA_CALLABLE_MEMBER CrossSection* getXSByZAID( int ZAID ) {
        CrossSection* ptr = nullptr;
        for( unsigned i = 0; i < size(); ++i ) {
            ptr = getXSPtr(i);
            if( ptr->ZAID() == ZAID ) {
                break;
            }
            ptr = nullptr;
        }
        return ptr;
    }

    CUDA_CALLABLE_MEMBER int size() const { return list_size; }

    CUDA_CALLABLE_MEMBER CrossSection* getListPtr() { return list; }
    CUDA_CALLABLE_MEMBER CrossSection& getXS(int i) { return list[i]; }
    CUDA_CALLABLE_MEMBER CrossSection* getXSPtr(int i) { return &(list[i]); }

private:

    managed_vector<CrossSection> list_vec;

    int list_size = 0;
    CrossSection* list = nullptr;
};

} /* namespace MonteRay */

#endif /* CROSSSECTIONLIST_HH_ */
