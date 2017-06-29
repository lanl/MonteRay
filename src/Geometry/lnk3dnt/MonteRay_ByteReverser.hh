#ifndef MONTERAY_BYTEREVERSER_HH_
#define MONTERAY_BYTEREVERSER_HH_

#include <cstring>
#include <type_traits>

namespace MonteRay {
  // Code: Pooma's MonteRay_reverseBytes.hh
  // Author: Peter McLachlan
  // The MonteRay_ByteReverser template defines the default behavior for
  // reversing bytes. It is templated on the type and on a bool that
  // indicates whether it is a basic "C" type or not.
  //
  //  -- If it is a basic C type, then the appropriate thing to do is
  //     to reverse the order of the sizeof(T) bytes in which the T
  //     object is stored.
  //  -- If it is not a basic C type, then we try to invoke the
  //     "MonteRay_reverseBytes" member function.
  // If a user's code fails to compile because he's using a T that
  // does not define T::MonteRay_reverseBytes(), he needs to write a
  // specialization of the global MonteRay_reverseBytes(T&) template for his
  // type.

  template <class T, bool basicType>
  struct MonteRay_ByteReverser;

  template <class T>
  struct MonteRay_ByteReverser<T,true>
  {
    inline static void MonteRay_reverseBytes(T &t)
    {
      T x;
      char *xb = reinterpret_cast<char*>(&x) + sizeof(T);
      char *b = reinterpret_cast<char*>(&t);
      for (unsigned int i = 0; i < sizeof(T); ++i)
        {
          *--xb = *b++;
        }

      t = x;
    }
  };

  template <class T>
  struct MonteRay_ByteReverser<T,false>
  {
    inline static void MonteRay_reverseBytes(T &t)
    {
      t.MonteRay_reverseBytes();
    }
  };

template <class T>
inline void MonteRay_reverseBytes(T &t)
{

    MonteRay_ByteReverser<T,std::is_pod<T>::value >::MonteRay_reverseBytes(t);
}

template <class T>
inline void MonteRay_reverseBytes( std::vector<T> &vec)
{
    for( unsigned i=0; i<vec.size(); ++i )
        MonteRay_ByteReverser<T,std::is_pod<T>::value>::MonteRay_reverseBytes( vec[i] );
}

inline void MonteRay_reverseBytes( char* cstring )
{
    for( unsigned i=0; i< std::strlen( cstring ); ++i )
        MonteRay_ByteReverser< char, true>::MonteRay_reverseBytes( cstring[i] );
}

} // end namespace MonteRay

#endif /* MONTERAY_BYTEREVERSER_HH_ */
