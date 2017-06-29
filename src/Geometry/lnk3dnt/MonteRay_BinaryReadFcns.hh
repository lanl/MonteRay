#ifndef MONTERAY_BINARY_READ_FCNS_HH
#define MONTERAY_BINARY_READ_FCNS_HH

#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>

#include "MonteRay_ByteReverser.hh"

namespace MonteRay {

class MonteRay_BinaryReader {
private:
    std::ifstream inFile;
    bool swapBytes;
    std::map<std::string,std::ifstream::pos_type> mymark;
public:
    MonteRay_BinaryReader( void ) : swapBytes( false ) {}
    MonteRay_BinaryReader( const std::string& );
    ~MonteRay_BinaryReader( void ) {
        if( inFile.is_open() )
            inFile.close();
    }
    void open( const std::string& );
    void VerifyStatus( void );
    
    template <typename T>
    void
    checkByteOrder( T& value, T expectedValue ) {
        int pos = inFile.tellg(); 
        inFile.read( reinterpret_cast<char*>( &value ), sizeof( value ) );
        inFile.seekg( pos );
        swapBytes = value != expectedValue;
    }
    template <typename T>
    void
    testByteOrder( T& value ) {
        int pos = inFile.tellg(); 
        inFile.read( reinterpret_cast<char*>( &value ), sizeof( value ) );
        inFile.seekg( pos );
    }
    void toggleByteSwapping( void ) { swapBytes = !swapBytes; }
    
    template<typename T> 
    void 
    read( std::vector<T>& value, int NumValues)
    {
        std::vector<T> temp(NumValues);
        inFile.read( reinterpret_cast<char*>( &temp[0] ), sizeof(T)*temp.size() );
        value.swap(temp);

        if( swapBytes )
        	MonteRay_reverseBytes( value );
    }
    
   template<typename T> 
    void
    read( T& value)
    {

	   if( !inFile.eof() ) {
           inFile.read( reinterpret_cast<char*>( &value ), sizeof( value ));
           if( swapBytes )
               MonteRay_reverseBytes( value );
	   } else {
		   throw std::logic_error("End of file reached");
	   }
	   if( inFile.fail() ) {
		   throw std::logic_error("Failed to read Value");
	   }
    }
    void read( std::string& );
    void 
    read( std::string&, unsigned );
    void read( char*, unsigned );
/// The Mark Functions will map a position in a file to a string.\n
/// These are very useful for the ReadLnk3dnt extract function where the file is randomly accessed.\n
    void setMark( const std::string & );
    void useMark( const std::string & );
    template< typename T >
    void advFile( const T off ) {
    	inFile.seekg( off, std::ios_base::cur);
    }

};

} // end namespace

#endif
