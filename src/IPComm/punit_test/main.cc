#include "MonteRay_unittest.hh"
#include "MonteRayMultiStream.hh"

#include <iostream>

#include <mpi.h>

int 
main( int argc, char* argv[] ) {

	MPI_Init(&argc, &argv);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MonteRay::MultiStream ParOut;
    if( world_rank == 0 ) {
        ParOut.setScreen();
    }
    UnitTest::TestReporterOstream<MonteRay::MultiStream> reporter( ParOut );

    unsigned int NLocalFailing = RunTests( argc, argv, reporter );

    unsigned int NGlobalFailing;
    
    MPI_Allreduce(&NLocalFailing, &NGlobalFailing, 1, MPI_UNSIGNED, MPI_SUM,
                  MPI_COMM_WORLD);

    MPI_Finalize();

    if( world_rank == 0 ) {
        return NGlobalFailing;
    } else {
        return 0;
    }

}
