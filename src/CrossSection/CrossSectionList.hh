#ifndef CROSSSECTIONLIST_HH_
#define CROSSSECTIONLIST_HH_

#include "CrossSection.hh"

namespace MonteRay {

template< typename HASHFUNCTION = FasterHash >
class CrossSectionList_t : public Managed {
public:
    using CrossSection = CrossSection_t<HASHFUNCTION>;

    CrossSectionList_t() = default;
    CrossSectionList_t(const CrossSectionList_t&) = delete;
    CrossSectionList_t(CrossSectionList_t&& other){
      this->list_vec = std::move(other.list_vec);
      this->list = other.list;
      this->list_size = other.list_size;
    }

    void add( CrossSection xs ) {
      auto checkZaid = [&] (const CrossSection& list_xs) { return xs.ZAID() == list_xs.ZAID(); };
      auto xsLoc = std::find_if(list_vec.begin(), list_vec.end(), checkZaid);
      if (xsLoc != list_vec.end()){ return; }

      xs.setID( list_vec.size() );
      list_vec.push_back(xs);

      list = list_vec.data();
      list_size = list_vec.size();
    }

    constexpr int size() const { return list_size; }
    constexpr CrossSection* getListPrtr() { return list; }
    constexpr CrossSection& getXS(int i) { return list[i]; }
    constexpr CrossSection* getXSPtr(int i) { return &(list[i]); }

private:

    managed_vector<CrossSection> list_vec;
    int list_size = 0;
    CrossSection* list = nullptr;
};

using CrossSectionList = CrossSectionList_t<>;

} /* namespace MonteRay */

#endif /* CROSSSECTIONLIST_HH_ */
