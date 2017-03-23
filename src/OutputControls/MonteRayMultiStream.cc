#include "MonteRayMultiStream.hh"

namespace MonteRay{

MultiStream::MultiStream( std::ostream* otherStream /* = NULL */ ) : screen( otherStream ) {}
MultiStream::MultiStream( std::ostream& otherStream ) : screen( &otherStream ) {}

void
MultiStream::clear( void ) {
    unsetScreen();
    outList.clear();
}

MultiStream&
MultiStream::operator<<( std::ostream& (*fcn)( std::ostream& ) )
{
    if( screen ) {
        (*fcn)( *screen );
    }
    for( const_iter i=outList.begin(); i!= outList.end(); ++i ) {
        (*fcn)( *(*i) );
    }
    return *this;
}

#ifdef __xlC__
MultiStream&
MultiStream::operator<<( std::ios_base& (*fcn)( std::ios_base& ) )
{
    if( screen ) {
        (*fcn)( *screen );
    }
    for( const_iter i=outList.begin(); i!= outList.end(); ++i ) {
        (*fcn)( *(*i) );
    }
    return *this;
}
#endif

} // end namespace
