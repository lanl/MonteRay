#include "mpi.h"
#include <stdio.h>

int main(int argc, char** argv) {

typedef double DATA_T;
DATA_T* ptrData;

MPI_Win shared_memory_window;
MPI_Comm shared_memory_communicator;
int shared_memory_size;
int shared_memory_rank;

MPI_Init(NULL,NULL);

MPI_Comm_split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared_memory_communicator);

MPI_Comm_rank( shared_memory_communicator, &shared_memory_rank );
MPI_Comm_size( shared_memory_communicator, &shared_memory_size );
if( shared_memory_rank == 0 ) {	printf( "Using=%d tasks\n", shared_memory_size ); }

MPI_Win_allocate_shared(10*sizeof(DATA_T),sizeof(DATA_T),MPI_INFO_NULL,shared_memory_communicator, &ptrData, &shared_memory_window);

// rank 0 initialises all data for all ranks
if(  shared_memory_rank == 0 ) {
    for( unsigned i=0; i < 10*shared_memory_size; ++i) {
        ptrData[i] = i;  
    }
}
MPI_Barrier( shared_memory_communicator );

// all ranks double their own data
for( unsigned i=0; i < 10; ++i) {
    ptrData[i] *= 2.0;  
}
MPI_Barrier( shared_memory_communicator );

// rank 0 sums the data
double total=0.0;
if(  shared_memory_rank == 0 ) {
    for( unsigned i=0; i < 10*shared_memory_size; ++i) {
        total += ptrData[i];  
    }
}

if( shared_memory_rank == 0 ) { printf("Answer should be 90 (1 proc), 380 (2 procs), 870 (3 procs), 1560 (4 procs)\n"); }
if( shared_memory_rank == 0 ) {	printf( "Total=%f\n", total ); }

MPI_Finalize();
return 0;
}

