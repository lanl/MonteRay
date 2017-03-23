#ifndef MONTERAYMULTISTREAM_HH_
#define MONTERAYMULTISTREAM_HH_

#include <memory>
#include <vector>
#include <fstream>
#include <iostream>

namespace MonteRay{

/// Class that handles multiple streams for Output. (From MCATK, but with C++ shared pointers and without boost::format)

///MultiStream can handle multiple streams of Output such as stdout, stderr, log files, and output file capturing of stdout/stderr. \n
///MultiStream also naturally handles all ostream manipulates specified in iomanip (i.e. setw, setprecision, etc. ).
///The use of Multistream can be found in the unit tests that comes with MultiStream. Multiple streams are handle with the use of \n
///the STL container called vector.
///
///This class also depends on one of the c++ smart pointers (shared pointer) to manage destruction of the \n
///pointer resources.
///
///The shared_ptr class template stores a pointer to a dynamically allocated object, typically with a C++ new-expression. \n
///The object pointed to is guaranteed to be deleted when the last shared_ptr pointing to it is destroyed or reset.
///
class MultiStream {
public:


    typedef std::ostream Stream_t;
    typedef std::shared_ptr< Stream_t > VecValue_t;

    ///OutList_t: A STL vector of type ostream shared pointers.
    typedef std::vector< VecValue_t > OutList_t;
    typedef OutList_t::const_iterator const_iter;

private:
    OutList_t outList;
    Stream_t* screen;

public:
    MultiStream( std::ostream* otherStream = NULL );
    MultiStream( std::ostream& otherStream );

    ///addFile: Lets user add a specifically named file stream.
    void addFile( const std::string& fileName, std::ios_base::openmode mode = std::ios_base::out ) {
        outList.push_back( std::make_shared<std::ofstream>(  fileName.c_str(), mode ) );
//        VecValue_t( new std::ofstream( fileName.c_str(), mode ))
    }

    ///addStream: Lets user add a stream.
    void addStream( VecValue_t pStream ) {
        outList.push_back( pStream );
    }

    ///addScreen: Lets user add a stream that goes to std::cout (stdout) by default.
    void setScreen( Stream_t& newScreen = std::cout){
            screen = &newScreen;
    }
    void clear( void );

    ///unsetScreen: Turns of the display to screen capability
    void unsetScreen( void ) {
            screen = NULL;
    }
    bool isScreenSet() const { return screen != NULL; }

    ///Templated operator to handle any type object for Multistream class
    template <typename T>
    MultiStream& operator<<( const T& someObj ) {
        if( screen )
            *screen << someObj;

        for( const_iter i=outList.begin(); i!= outList.end(); ++i )
            *(*i) << someObj;

        return *this;
    }

    /// Non-templated operator version that accepts std::endl, std::ends, std::flush
    MultiStream& operator<<( std::ostream& (*manipulator)( std::ostream& ) );
#ifdef __xlC__
    MultiStream& operator<<( std::ios_base& (*manipulator)( std::ios_base& ) );
#endif

    bool isActive( void ) { return !outList.empty() || isScreenSet(); }
};

} // end namespace
#endif /* MULTISTREAM_HH_ */
