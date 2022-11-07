#include <stdio.h>
#include "mpi.h"
#include "wsp-mpi.h"

int main(int argc, char *argv[])
{
    int procID;
    int procNum;
    double startTime;
    double endTime;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &procID);

    // Get total number of processes specificed at start of run
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);

    parseArgs(argc, argv);

    // Run computation
    startTime = MPI_Wtime();
    wspStart(procID, procNum);
    endTime = MPI_Wtime();

    // Compute running time
    MPI_Finalize();

    double time = endTime - startTime;
    printf("elapsed time for proc %d: %.3lf ms (%.3lf s)\n", procID, time * 1000., time);

    wsp_print_result(procID, time);

    freePath();
    return 0;
}