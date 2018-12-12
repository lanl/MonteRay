#include "needebugger.hh"

#include <string>
#include <iostream>

int main(int argc, char *argv[]) {

    using namespace nee_debugger_app;

    if( argc < 2 ) {
        std::cout << "ERROR: You must supply a base name as an argument\n";
    }

    std::string filename = std::string( argv[1] );

    nee_debugger nee;
    nee.launch( filename );

}
