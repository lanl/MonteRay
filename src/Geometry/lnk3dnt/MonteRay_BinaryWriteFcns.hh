#ifndef MONTERAY_BINARY_WRITE_FCNS_HH
#define MONTERAY_BINARY_WRITE_FCNS_HH

#include <fstream>
#include <string>
#include <vector>

namespace MonteRay {

class MonteRay_BinaryWriter {
    
private:
    std::string FileName;
    std::ofstream outFile;
    void OpenToAppend() { 
        outFile.open( name(), std::ios::binary | std::ios::app ); 
    }
public:
    MonteRay_BinaryWriter( const std::string& filename ) :
        FileName( filename ),
        outFile( name(), std::ios::binary | std::ios::trunc )
        {}
    ~MonteRay_BinaryWriter( void ) {
        if( outFile.is_open() )
            outFile.close(); 
    }
    const char* name( void ) { return FileName.c_str(); }
    
    /// Function allows quick insertion that is immediately followed by closing file
    template< typename T >
    void append( const T& value ) {
        if( !outFile.is_open() ) OpenToAppend();
        write( value );
        outFile.close();
    }
    
    /// Function allows quick insertion that is immediately followed by closing file
    template< typename T >
    void append( const T& value, unsigned N ) {
        if( !outFile.is_open() ) OpenToAppend();
        write( value, N );
        outFile.close();
    }
    
    template<typename T> 
    void 
    write( const std::vector<T>& value )
    {
        outFile.write( reinterpret_cast<const char*>( &value[0] ), sizeof(T)*value.size() );
    }
    
    /// This version is probably unsafe since the size of value is not being
    /// checked before reading from it.
    template<typename T> 
    void 
    write( const T* value, int N)
    {
        outFile.write( reinterpret_cast<const char*>( value ), sizeof(T)* N );
    }

    template<typename T> 
    void
    write( const T& value)
    {
        outFile.write( reinterpret_cast<const char*>( &value ), sizeof( value ));
    }

    void write( const std::string& s );
    
    void 
    write( const std::string& s, unsigned N ) {
        //if s.size() is less than N, pad s with spaces
        outFile.write( s.c_str(), N );
    }

};

}


#endif
