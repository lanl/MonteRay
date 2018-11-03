#include "mpi.h"
#include <stdio.h>

int main(int argc, char** argv) {

/*
 Tester that ensures proper functioning of MPI3 Shared Memory
 - Shows issue with IBM Spectrum MPI for > 1 process

 Jeremy Sweezy - jsweezy@lanl.gov
*/

typedef double DATA_T;
DATA_T* ptrData1;
DATA_T* ptrData2;

MPI_Win shared_memory_window1;
MPI_Win shared_memory_window2;

MPI_Comm shared_memory_communicator;

int shared_memory_size;
int shared_memory_rank;

MPI_Init(NULL,NULL);

MPI_Comm_split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared_memory_communicator);

MPI_Comm_rank( shared_memory_communicator, &shared_memory_rank );
MPI_Comm_size( shared_memory_communicator, &shared_memory_size );
if( shared_memory_rank == 0 ) { printf( "Using=%d tasks\n", shared_memory_size ); }

MPI_Win_allocate_shared(10*sizeof(DATA_T),sizeof(DATA_T),MPI_INFO_NULL,shared_memory_communicator, &ptrData1, &shared_memory_window1);

// allocation of second shared memory window exposes bug in IBM Spectrum MPI
MPI_Win_allocate_shared(10*sizeof(DATA_T),sizeof(DATA_T),MPI_INFO_NULL,shared_memory_communicator, &ptrData2, &shared_memory_window2);

MPI_Barrier( shared_memory_communicator );

// rank 0 initialises all data for all ranks
if(  shared_memory_rank == 0 ) {
    for( unsigned i=0; i < 10*shared_memory_size; ++i) {
        ptrData1[i] = i;
    }

    // fill up ptrData -- shouldn't affect ptrData1 values
    for( unsigned i=0; i < 10*shared_memory_size; ++i) {
        ptrData2[i] = 10000+i;
    }

    // test proper values in ptrData1
    bool failure = false;
    for( unsigned i=0; i < 10*shared_memory_size; ++i) {
       if( ptrData1[i] != i ) {
          printf("ERROR:  i=%u,  ptrData1[%u] is %f but should be %f !!\n", i, i, ptrData1[i], DATA_T( i ) );
          failure = true;
       }
    }
    if( failure ) {
        printf("\n");
        printf("ERROR: ------------------------------------ \n");
        printf("ERROR: --------- Failure Diagnostics ------ \n");
        printf("ERROR: pointer value of ptrData1 = %p, %u \n", ptrData1, ptrData1);
        printf("ERROR: pointer value of ptrData2 = %p, %u \n", ptrData2, ptrData2);
        printf("ERROR: ------------------------------------ \n");
        printf("\n");
    }
}
MPI_Barrier( shared_memory_communicator );

// all ranks double their own data
for( unsigned i=0; i < 10; ++i) {
    // double ptrData1 but not ptrData2
    ptrData1[i] *= 2.0;
}
MPI_Barrier( shared_memory_communicator );

// rank 0 sums the data
double total=0.0;
if(  shared_memory_rank == 0 ) {
    for( unsigned i=0; i < 10*shared_memory_size; ++i) {
        // should be unaffected by ptrData2
        total += ptrData1[i];
    }
}

DATA_T expected_value = 0.0;
for( unsigned i=0; i < 10*shared_memory_size; ++i ) {
    expected_value += 2.0*i;
}

if( shared_memory_rank == 0 ) {
    // if( total < (expected[shared_memory_size-1]-1.0e-14 ) or total > (expected[shared_memory_size-1]+1.0e-14 ) ) {
    if( total < (expected_value-1.0e-14 ) or total > (expected_value+1.0e-14 ) ) {
        printf("FAILURE: total was %f but should be %f !! \n", total, expected_value );
    } else {
        printf("PASS: total is correct, %f .\n", total );
    }
}

MPI_Finalize();
return 0;
}
