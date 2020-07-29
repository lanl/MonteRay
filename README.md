MonteRay 
----------------

MonteRay is a library for providing neutron particle ray-casting based Monte Carlo tallies.  MonteRay can provide ray-casting based tallies on CPU and GPU hardware.  MonteRay can provide significant performance benefits of GPU hardware.  MonteRay is intended for use with un-accelerating Monte Carlo codes running on CPU hardware.  

To clone MonteRay:

    $ git clone https://github.com/lanl/MonteRay.git


MonteRay Dependencies
---------------------
* CMake 3.17
* C++ compiler (GCC 6.4.0)
* CUDA (Optional acceleration on NVIDIA based GPUs )
* MPI-3 capabile MPI implementation (OpenMPI, IBM Spectrum MPI)
* UnitTest++ -- https://github.com/unittest-cpp/unittest-cpp
* MPark.Variant -- https://github.com/mpark/variant
