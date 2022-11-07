#include <stdio.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include "mpi.h"
#include "wsp-mpi.h"

path_t *bestPath = NULL;

int NCITIES = -1; // number of cities in the file.
int *DIST = NULL; // one dimensional array used as a matrix of size (NCITIES * NCITIES).

// set DIST[i,j] to value
inline static void set_dist(int i, int j, int value)
{
  assert(value > 0);
  int offset = i * NCITIES + j;
  DIST[offset] = value;
  return;
}

// returns value at DIST[i,j]
inline static int get_dist(int i, int j)
{
  int offset = i * NCITIES + j;
  return DIST[offset];
}

// prints results
void wsp_print_result(int procID, double time)
{
    if (procID != 0)
        return;

    printf("========== Solution ==========\n");
    printf("Cost: %d\n", bestPath->cost);
    printf("Path: ");
    for(int i = 0; i < NCITIES; i++) {
        if(i == NCITIES-1) printf("%d", bestPath->path[i]);
        else printf("%d -> ", bestPath->path[i]);
    }

    printf("\n\n============ Time ============\n");
    printf("Time: %.3f ms (%.3f s)\n\n", time * 1000., time);
}

void parseArgs(int argc, char *argv[])
{
    if(argc < 2) error_exit("Expecting one argument: [file name]\n");

    char *filename = argv[1];
    FILE *fp = fopen(filename, "r");
    if(fp == NULL) error_exit("Failed to open input file \"%s\"\n", filename);
    int scan_ret;
    scan_ret = fscanf(fp, "%d", &NCITIES);
    if(scan_ret != 1) error_exit("Failed to read city count\n");
    if(NCITIES < 2) {
        error_exit("Illegal city count: %d\n", NCITIES);
    } 
    // Allocate memory and read the matrix
    DIST = (int*)calloc(NCITIES * NCITIES, sizeof(int));
    SYSEXPECT(DIST != NULL);
    for(int i = 1;i < NCITIES;i++) {
        for(int j = 0;j < i;j++) {
            int t;
            scan_ret = fscanf(fp, "%d", &t);
            if(scan_ret != 1) error_exit("Failed to read dist(%d, %d)\n", i, j);
            set_dist(i, j, t);
            set_dist(j, i, t);
        }
    }
    fclose(fp);
    bestPath = (path_t*)malloc(sizeof(path_t));
    bestPath->cost = 0;
    bestPath->path = (city_t*)calloc(NCITIES, sizeof(city_t));
}

static void collect(int procID, int procNum)
{
    if (procID != 0)
    {
        MPI_Send(&bestPath->cost, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(bestPath->path, NCITIES, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }
    else
    {
        int currCost;
        MPI_Status status;
        for (int i = 1; i < procNum; i++)
        {
            MPI_Recv(&currCost, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            if (currCost < bestPath->cost)
            {
                bestPath->cost = currCost;
                MPI_Recv(bestPath->path, NCITIES, MPI_CHAR, i, 1, MPI_COMM_WORLD, &status);
            }
        }
    }
}

void wspStart(int procID, int procNum)
{
    if (procID > NCITIES)
        return;

    bestPath->cost = 0x7FFFFFFF; // 2^31 - 1

    for (int initID = procID; initID < NCITIES; initID += procNum)
    {
        int currCost = 0;
        std::vector<int> currPath(NCITIES);

        currPath[0] = initID;
        for (int i = 1; i < NCITIES; i++)
        {
            currPath[i] = (currPath[i - 1] + 1) % NCITIES;
            currCost += get_dist(currPath[i - 1], currPath[i]);
        }

        if (currCost < bestPath->cost)
        {
            std::copy(currPath.begin(), currPath.end(), bestPath->path);
            bestPath->cost = currCost;
        }

        while(std::next_permutation(currPath.begin() + 1, currPath.end()))
        {
            currCost = 0;
            int i;
            for(i = 1; i < NCITIES; i++)
            {
                currCost += get_dist(currPath[i - 1], currPath[i]);
                if (currCost >= bestPath->cost)
                    break;
            }
            if (i == NCITIES)
            {
                std::copy(currPath.begin(), currPath.end(), bestPath->path);
                bestPath->cost = currCost;
            }
        }
    }
    collect(procID, procNum);
}

void freePath()
{
    free(bestPath->path);
    free(bestPath);
}