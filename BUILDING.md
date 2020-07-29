## Building MonteRay

### Prerequisites
* CMake 3.17 or later
* C++ compiler (GCC 6.4.0 or later)
* MPI implmentation that supports MPI-3 (OpenMPI 1.10 or later )
* UnitTest++ -- https://github.com/unittest-cpp/unittest-cpp
* MPark.Variant -- https://github.com/mpark/variant
* MonteRay Test Files -- https://github.com/lanl/MonteRayTestFiles

### Configuring your environment
Set environment variables for package location (examples here are for tcsh)

* MonteRay test file location
     setenv MONTERAY_TESTFILES_DIR  /home/username/MonteRayTestFiles

* UnitTest++ installation location
     setenv UNITTEST_ROOT /home/username/packages/UnitTest++

* MPark Variant
      setenv CMAKE_PREFIX_PATH /home/username/packages/variant

* MPI path
      setenv PATH /home/username/packages/openmpi-3.1.3/bin:$PATH
      setenv LD_LIBRARY_PATH /home/username/packages/openmpi-3.1.3/lib:$LD_LIBRARY_PATH

### Configuring CMake
* Use a separate build director

    mkdir build
    cd build
      
*To configure with CUDA enabled:
    cmake -Denable_cuda=ON ..

*To configure without CUDA enabled (CPU only):
    cmake ..

* Build
    make [-j [N]]

* Test 
    ctest

     
