#ifndef MONTERAY_BINARYIO_HH_
#define MONTERAY_BINARYIO_HH_

#include <iostream>
#include <stdexcept>
#include <cassert>

namespace MonteRay{

namespace binaryIO{

// Determine if a type is a pointer
// Used to verify that the correct write function is called when writing dynamically allocated arrays
template <typename T>
struct isPointer { static const bool value = false; };
template <typename T>
struct isPointer<T*> { static const bool value = true; };

// Write scalars and statically allocated arrays
template<typename S, typename T>
void
write( S& outFile, const T& value)
{
#if !defined(CUDA)
    static_assert( !isPointer<T>::value, "Dynamic arrays must call 3 argument version of binaryIO::write.\n" );
#endif
    outFile.write( reinterpret_cast<const char*>( &value ), sizeof( value ));
    if( outFile.fail() ) {
        throw std::logic_error("binaryIO::write -- Failed to write Value");
    }
}

// Write dynamically allocated arrays
template<typename S, typename T>
void
write( S& outFile, const T* value, unsigned N)
{
    outFile.write( reinterpret_cast<const char*>( &N ), sizeof(N));
    outFile.write( reinterpret_cast<const char*>( value ), N*sizeof(T));
    if( outFile.fail() ) {
        throw std::logic_error("binaryIO::write -- Failed to write Value");
    }
}

// Write statically allocated arrays
// For Intel and gnu compilers, statically allocated arrays go to write( S& outFile, const T& value)
template<typename S, typename T, unsigned N>
void write( S& outFile, const T(&value)[N] )
{
    outFile.write( reinterpret_cast<const char*>( value ), N*sizeof(T));
    if( outFile.fail() ) {
        throw std::logic_error("binaryIO::write -- Failed to write Value");
    }
}

// Write strings
template<typename S>
void
write( S& outFile, const std::string& value)
{
    // Convert the string to char* and call the standard array write function
    write( outFile, value.c_str(), value.size()+1 );
}

// Read scalars and statically allocated arrays
template<typename S, typename T>
void
read( S& inFile, T& value)
{
    if( !inFile.eof() ) {
        inFile.read( reinterpret_cast<char*>( &value ), sizeof( value ));
    } else {
        throw std::logic_error("binaryIO::read -- End of file reached");
    }
    if( inFile.fail() ) {
        throw std::logic_error("binaryIO::read -- Failed to read Value");
    }
}

// Read dynamically allocated arrays
template<typename S, typename T>
void
read( S& inFile, T*& value, unsigned N=0 )
{
    if( !inFile.eof() ) {
        unsigned Size;
        inFile.read( reinterpret_cast<char*>( &Size ), sizeof(Size) );
        // If the size is unknown prior to reading, allocate space and memory and point value to the newly allocated space
        if( N == 0 ){
            value = new T [Size];
            // If the size is known, but does not match what is located in the file, do not attempt to read the data
        } else if( Size != N ) {
            throw std::logic_error("binaryIO::read -- Expected array size does not match actual array size in file");
        }

        inFile.read( reinterpret_cast<char*>( value ), Size*sizeof(T));

        // End of file reached
    } else {
        throw std::logic_error("binaryIO::read -- End of file reached");
    }
    if( inFile.fail() ) {
        throw std::logic_error("binaryIO::read -- Failed to read Value");
    }
}


// Read statically allocated arrays
// For Intel and gnu compilers, statically allocated arrays go to read( S& outFile, T& value)
template<typename S, typename T, unsigned N>
void read( S& inFile, T(&value)[N] )
{
    if( !inFile.eof() ) {
        inFile.read( reinterpret_cast<char*>( value ), N*sizeof(T));
    } else {
        throw std::logic_error("binaryIO::read -- End of file reached");
    }
    if( inFile.fail() ) {
        throw std::logic_error("binaryIO::read -- Failed to read Value");
    }
}

// Read strings
template<typename S>
void
read( S& inFile, std::string& value)
{
    char* buffer = 0;
    read( inFile, buffer );
    value = buffer;
}

}

}
#endif /* BINARYIO_HH_ */
