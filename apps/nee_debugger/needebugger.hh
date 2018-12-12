#ifndef NEEDEBUGGER_HH_
#define NEEDEBUGGER_HH_

#include <string>

namespace nee_debugger_app {

class nee_debugger {
public:
    nee_debugger(){};

    void launch(const std::string& optBaseName);

    ~nee_debugger(){};

    void checkFileExists(const std::string& filename);
};

} /* namespace nee_debugger */

#endif /* NEEDEBUGGER_HH_ */

