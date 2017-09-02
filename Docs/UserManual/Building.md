Building MonteRay      {#building}
================================================================== 

Build System Overview     {#BuildSystem}
========

The MonteRay CMake build system is borrowed from MCATK.  Monte Ray requires at a mimimum CMake 3.7 
to properly support CUDA. On systems that support MCATK (LANL HPC machines, LANL X-Div Network, LANL's 
Darwin cluster) dependencies are automatically located with CMake.  On other systems the user must 
provide the locations through environment variables.  

Required Software  {#RequiredSoftware}
========

* CMake (version 3.7 minimum )
* Doxygen ( for creating documentation )
* UnitTest++ 
* MCATK LNK3DNT Test Files
* MonteRay Binary Test Files

Building MonteRay  {#BuildingMonteRay}
========

We will first cover building MonteRay on standard LANL systems, then for generic stand-alone systems.
Standard LANL systems have the required software which is auto-located with CMake.  
MonteRay, like MCATK, is not built in the standard CMake way, but using a CMake script, `Builder.cmake`.   
MonteRay can be built without this script but results may vary.  _Be cautious use the Builder script._

The first thing the user must do is set the __BINARY_DIR__ environment variable.  This variable tells
the Builder script where to build MonteRay.  The __BINARY_DIR__ environment variable also indicates 
the compiler and debug build level.  The form is __COMPILER-DEBUGLEVEL-NAME__.    For example to build
MonteRay with the GNU compiler in Release mode the form would be:

    export BINARY_DIR=/home/user1/obj/gnu-Release-MonteRay 

The options for the COMPILER portion of the name are: gnu, intel,  xl, or clang.  

The options for the DEBUGLEVEL portion of the name are: debug, db, Debug, DEBUG, release, Release, 
RELEASE.  If neither debug or release is specified the default compile is "Release with Debug Info".

Note that the COMPILER and DEBUGLEVEL options are taken from the BINARY_DIR environment variable using
a simple regular expression match.   So be careful when specifying the BINARY_DIR. The following example 
will not use the clang compiler in release mod.

    export BINARY_DIR=/home/user1/obj/clang-Release-FoxlikeBedbugs

Instead `xl` will be matched in Foxlike and `db` will be matched in Bedbugs, resulting in a debug 
compile using the IBM XL compiler.

Simplest Build Procedure {#SimplestBuildProcedure}
-----------------

The simplest build procedure on LANL systems is excuted from the base MonteRay directory after setting 
the BINARY_DIR :

    export BINARY_DIR=/home/user1/obj/gnu-Relelase-MonteRay
    cd /home/user1/MonteRaySource
    cmake -P cmake_files/Builder.cmake

Then to build MonteRay change to the binary directory and execute `make all` or `make install`.

    cd $BINARY_DIR
    make -j 16 install

Building can also be accomplished within Nsight or Eclipse.   In Eclipse, set the BINARY_DIR environment 
variable by adding it under Project->Properties->Build->Environment.  Cmake is executed with the make 
target in Eclipse named "Initiate Build".  Then build with the install target, or using any individual 
unit test build target. 

Building on Standalone systems {#BuildingOnStandAlone}
-----------------

To build on standalone systems, with the required software located in non-MCATK standard paths, the user must 
pass the locations via environment variables.  

### BINARY_DIR {#StandAloneBINARYDIR} ###

Set the BINARY_DIR variable as described above.

### UnitTest++ {#StandAloneUnitTest} ###

The MCATK team archives a copy of UnitTest++ with a custom CMake build system.  It is stored in svn
and can be checkout with: 
  
    svn checkout svn+ssh://username@xlogin/home/xshares/PROJECTS/mcatk/svn/repo/ThirdParty/UnitTest++
  
Remove any dashes (`-`) added by LaTeX in the above path. 
If an installation of UnitTest++ can not be located in a standard MCATK tools location, the path 
can be specified with an environment variable, __UNITTEST_DIR__.  For example
 
    export UNITTEST_DIR=/home/user1/packages/UnitTest++

### Required Testing files {#TestingFiles} ###

MonteRay is developed with Test Driven Development (TDD).  Thus the tests are run during ( not 
following ) a standard compilation.  However, testing can be turned off or delayed to after final 
compilation.  To perform the standard unit testing the MonteRay build system needs to find the location
of two directories. One directory contains LNK3DNT geometry files.  Another directory contains 
cross-section files, collision point files, and tally data files.  These files are used to test 
MonteRay in the absence of a Monte Carlo transport code.  Some of the collision point files have 
been generated using real-world examples via MCATK, thus allowing the performance of MonteRay to be 
quickly evaluated on different hardware or with different algorithms.

__MonteRay testing files__

MonteRay binary cross-section files, collision point files, and tally files copied from:
 
    /home/xshares/PROJECTS/mcatk/MonteRayTestFiles

This directory is located automatically by CMake, however to specify a non-standard location set the
__MONTERAY_TESTFILES_DIR__ environment variable.  Example:

    export MONTERAY_TESTFILES_DIR=/home/user1/MonteRayTestFiles

__LNK3DNT geometry files__

LNK3DNT geometry files are copied from the master location on the X-Div network located at:

    /home/xshares/PROJECTS/mcatk/lnk3dnt

This directory is located automatically by CMake, however to specify a non-standard location set the
__LNK3DNTDIR__ environment variable.  Example:

    export LNK3DNTDIR=/home/user1/lnk3dnt

### Other Environment Variables {#OtherEnvVariables} ###

__INSTALLDIR__

The __INSTALLDIR__ environment variable specifies where the binary library is to be be installed. 
This variable is required for a stand-alone build.

__PACKAGEDIR__

The __PACKAGEDIR__ environment variable is not required if UnitTest++ is found with the ++UNITTEST_DIR.
In MCATK this variable is used to find the location of multiple packages (Boost, Loki, and UnitTest++).  
In MonteRay can be used as an alternative method for locating UnitTest++, but it is not necessary.

### Running CMake for Standalone Systems {#BuildingStandalone}

To build MonteRay on standalone systems the standalone flag must be passed to CMake.
 
    cmake -DStandalone:BOOL=ON -P cmake_files/Builder.cmake
    
Combining this with the necessary environment variables (in tcsh this time):   

    setenv BINARY_DIR /g/g17/jsweezy/cuda-workspace/MonteRay_git_LLNL/gnu-release-build
    setenv LNK3DNTDIR /g/g17/jsweezy/lnk3dnt
    setenv MONTERAY_TESTFILES_DIR /g/g17/jsweezy/MonteRayTestFiles
    setenv UNITTEST_DIR /g/g17/jsweezy/packages/UnitTest++
    setenv INSTALLDIR /g/g17/jsweezy/MonteRay-release
    cmake -DStandalone:BOOL=ON -P cmake_files/Builder.cmake
    cd $BINARY_DIR
    make -j 20 all
 







