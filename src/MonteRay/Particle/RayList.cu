#include "RayList.hh"
#include "MonteRayParallelAssistant.hh"

namespace MonteRay {

template<unsigned N>
void
RayList_t<N>::writeToFile( const std::string& filename) const {
    std::ofstream out;
    out.open( filename.c_str(), std::ios::binary | std::ios::out);
    write( out );
    out.close();
}

template<unsigned N>
void
RayList_t<N>::readFromFile( const std::string& filename) {
    std::ifstream in;
    in.open( filename.c_str(), std::ios::binary | std::ios::in);
    if( ! in.good() ) {
        throw std::runtime_error( "RayList::readFromFile -- can't open file for reading" );
    }
    read( in );
    in.close();
}

template class RayList_t<1>;
template class RayList_t<3>;

} // end namespace


