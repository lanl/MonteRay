#ifndef CROSSSECTIONLIST_HH_
#define CROSSSECTIONLIST_HH_

#include "CrossSection.hh"

namespace MonteRay {

template< typename HASHFUNCTION = FasterHash >
class CrossSectionList_t : public Managed {
public:
    CrossSectionList_t(){};

    ~CrossSectionList_t(){};

    void add( CrossSection_t<HASHFUNCTION> xs ) {
        CrossSection_t<HASHFUNCTION>* ptr = getXSByZAID( xs.ZAID() );
        if( ptr ) { return; }

        xs.setID( list_vec.size() );
        list_vec.push_back(xs);

        list = list_vec.data();
        list_size = list_vec.size();
    }

    CUDA_CALLABLE_MEMBER CrossSection_t<HASHFUNCTION>* getXSByZAID( int ZAID ) {
        CrossSection_t<HASHFUNCTION>* ptr = nullptr;
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

    CUDA_CALLABLE_MEMBER CrossSection_t<HASHFUNCTION>* getListPtr() { return list; }
    CUDA_CALLABLE_MEMBER CrossSection_t<HASHFUNCTION>& getXS(int i) { return list[i]; }
    CUDA_CALLABLE_MEMBER CrossSection_t<HASHFUNCTION>* getXSPtr(int i) { return &(list[i]); }

private:

    managed_vector<CrossSection_t<HASHFUNCTION>> list_vec;

    int list_size = 0;
    CrossSection_t<HASHFUNCTION>* list = nullptr;
};

using CrossSectionList = CrossSectionList_t<>;

} /* namespace MonteRay */

#endif /* CROSSSECTIONLIST_HH_ */
