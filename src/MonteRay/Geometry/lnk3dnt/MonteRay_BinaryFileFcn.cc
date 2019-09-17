#include "MonteRay_BinaryReadFcns.hh"
#include "MonteRay_BinaryWriteFcns.hh"
#include <stdexcept>
#include <sstream>

#include <cstdlib>

namespace MonteRay {

using namespace std;

MonteRay_BinaryReader::MonteRay_BinaryReader( const string& filename )
{
    open( filename );
}

void
MonteRay_BinaryReader::open( const std::string& filename )
{
    inFile.open( filename.c_str(), ios::binary ),
    swapBytes = false;
    try {

        VerifyStatus();

    } catch( std::exception& err ) {

        stringstream ErrMsg;
        ErrMsg << "Error: Unable to open file **"<<filename<<"**."<<endl;
        if( filename[ 0 ] != '/' ) {
              ErrMsg << "Looking in directory : "<<getenv( "PWD") << endl;
        }
        ErrMsg << "Detected in : "<< __FILE__ << "["<<__LINE__<<"] : "<< "MonteRay_BinaryReader::open" <<endl;
        throw std::runtime_error( ErrMsg.str() );
    }

}

void
MonteRay_BinaryReader::VerifyStatus( void ) {
    if( !inFile.is_open() || !inFile ) {
        throw std::exception();
    }
}

void
MonteRay_BinaryReader::read( string& str, unsigned size ) {
    char Buffer[ size + 1 ];
    for( unsigned i=0; i<size+1; ++i )
        Buffer[ i ] = '\0';

    inFile.read( Buffer, size );
    if( swapBytes ) {
        MonteRay_reverseBytes( Buffer );
    }

    str = Buffer;
}

void
MonteRay_BinaryReader::setMark( const std::string &name ) {
    mymark[name]=inFile.tellg();
}
void
MonteRay_BinaryReader::useMark( const std::string &name ) {
    if(mymark.find(name) == mymark.end() ) {
        std::stringstream msg;
        msg  << "ERROR:  Bookmark * "<< name <<" * was not found!\n";
        throw std::logic_error( msg.str() );
    }
    inFile.seekg(mymark[name]);
}

}
