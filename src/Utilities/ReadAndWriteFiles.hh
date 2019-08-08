#ifndef MR_READANDWRITEFILES_HH_
#define MR_READANDWRITEFILES_HH_

namespace MonteRay{

template <class Object>
auto readFromFile( const std::string& filename, Object& object) {
  std::ifstream infile;
  if( infile.is_open() ) {
      infile.close();
  }
  infile.open( filename.c_str(), std::ios::binary | std::ios::in);

  if( ! infile.is_open() ) {
      fprintf(stderr, "Error:  readFromFile -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
      throw std::runtime_error("readFromFile -- Failure to open file" );
  }
  assert( infile.good() );
  infile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
  auto retval = object.read(infile);
  infile.close();
  return retval;
}

template <class Object>
void writeToFile( const std::string& filename, Object&& object) {
  std::ofstream outfile;
  outfile.open( filename.c_str(), std::ios::binary | std::ios::out);
  if( ! outfile.is_open() ) {
    fprintf(stderr, "writeToFile -- Failure to open file,  filename=%s  %s %d\n", filename.c_str(), __FILE__, __LINE__);
    throw std::runtime_error("writeToFile -- Failure to open file" );
  }
  assert( outfile.good() );
  outfile.exceptions(std::ios_base::failbit | std::ios_base::badbit );
  object.write( outfile );
  outfile.close();
}

}
#endif
