#include "framework.h"

#include <mpi.h>

int ParallelFramework::getNumOfProcesses() const {
    int num;
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    return num;
}

int ParallelFramework::receiveRequest(int& source) const {
    MPI_Status status;
    MMPI_Recv(nullptr, 0, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    source = status.MPI_SOURCE;
    return status.MPI_TAG;
}

unsigned long ParallelFramework::receiveMaxBatchSize(int mpiSource) const {
    MPI_Status status;
    unsigned long maxBatchSize;
    MMPI_Recv(&maxBatchSize, 1, MPI_UNSIGNED_LONG, mpiSource, TAG_MAX_DATA_COUNT, MPI_COMM_WORLD, &status);
    return maxBatchSize;
}

void ParallelFramework::sendBatchSize(const AssignedWork& work, int mpiSource) const {
    MPI_Send(&work, 2, MPI_UNSIGNED_LONG, mpiSource, TAG_DATA, MPI_COMM_WORLD);
}

void ParallelFramework::receiveAllResults(float *dst, size_t count, int mpiSource) const {
    MPI_Status status;
    MMPI_Recv(dst, count, RESULT_MPI_TYPE, mpiSource, TAG_RESULTS_DATA, MPI_COMM_WORLD, &status);
}

int ParallelFramework::receiveListResults(double *dst, size_t maxCount, int mpiSource) const {
    MPI_Status status;
    MMPI_Recv(dst, parameters.overrideMemoryRestrictions ? INT_MAX : maxCount * parameters.D, DATA_MPI_TYPE, mpiSource, TAG_RESULTS_DATA, MPI_COMM_WORLD, &status);

    // Find the number of points in list
    int receivedCount;
    MPI_Get_count(&status, DATA_MPI_TYPE, &receivedCount);

    // MPI_Get_count returned the count of DATA_TYPE elements received, so divide with D to get the count of points
    return receivedCount / parameters.D;
}

void ParallelFramework::syncWithSlaves() const {
    // MPI_Barrier(MPI_COMM_WORLD);
    int a = 0;
    MPI_Bcast(&a, 1, MPI_INT, 0, MPI_COMM_WORLD);
}
