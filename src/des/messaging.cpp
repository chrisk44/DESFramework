#include "desf.h"

#include "definitions.h"
#include "utilities.h"

#include <mpi.h>

int DesFramework::getNumOfProcesses() const {
    int num;
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    return num;
}

void DesFramework::sendReadyRequest(unsigned long maxBatchSize) const {
    MPI_Send(nullptr, 0, MPI_INT, 0, TAG_READY, MPI_COMM_WORLD);
    MPI_Send(&maxBatchSize, 1, MPI_UNSIGNED_LONG, 0, TAG_MAX_DATA_COUNT, MPI_COMM_WORLD);
}

void DesFramework::sendBatchSize(const AssignedWork& work, int mpiSource) const {
    MPI_Send(&work, 2, MPI_UNSIGNED_LONG, mpiSource, TAG_DATA, MPI_COMM_WORLD);
}

int DesFramework::receiveRequest(int& source) const {
    MPI_Status status;
    MMPI_Recv(nullptr, 0, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    source = status.MPI_SOURCE;
    return status.MPI_TAG;
}

unsigned long DesFramework::receiveMaxBatchSize(int mpiSource) const {
    MPI_Status status;
    unsigned long maxBatchSize;
    MMPI_Recv(&maxBatchSize, 1, MPI_UNSIGNED_LONG, mpiSource, TAG_MAX_DATA_COUNT, MPI_COMM_WORLD, &status);
    return maxBatchSize;
}

void DesFramework::sendExitSignal() const {
    MPI_Send(nullptr, 0, MPI_INT, 0, TAG_EXITING, MPI_COMM_WORLD);
}

AssignedWork DesFramework::receiveWorkFromMaster() const {
    AssignedWork work;
    MMPI_Recv(&work, 2, MPI_UNSIGNED_LONG, 0, TAG_DATA, MPI_COMM_WORLD, nullptr);
    return work;
}

void DesFramework::sendResults(RESULT_TYPE *data, size_t count) const {
    MPI_Send(nullptr, 0, MPI_INT, 0, TAG_RESULTS, MPI_COMM_WORLD);
    MPI_Send(data, count, RESULT_MPI_TYPE, 0, TAG_RESULTS_DATA, MPI_COMM_WORLD);
}

void DesFramework::receiveAllResults(RESULT_TYPE *dst, size_t count, int mpiSource) const {
    MPI_Status status;
    MMPI_Recv(dst, count, RESULT_MPI_TYPE, mpiSource, TAG_RESULTS_DATA, MPI_COMM_WORLD, &status);
}

void DesFramework::sendListResults(DATA_TYPE *data, size_t numOfPoints) const {
    MPI_Send(nullptr, 0, MPI_INT, 0, TAG_RESULTS, MPI_COMM_WORLD);

    MPI_Send(&numOfPoints, 1, MPI_UNSIGNED_LONG, 0, TAG_RESULTS_COUNT, MPI_COMM_WORLD);
    MPI_Send(data, numOfPoints * m_config.model.D, DATA_MPI_TYPE, 0, TAG_RESULTS_DATA, MPI_COMM_WORLD);
}

int DesFramework::receiveListResults(std::vector<DATA_TYPE>& dst, size_t maxCount, int mpiSource) const {
    MPI_Status status;

    size_t count;
    MMPI_Recv(&count, 1, MPI_UNSIGNED_LONG, mpiSource, TAG_RESULTS_COUNT, MPI_COMM_WORLD, &status);

    if(count > maxCount)
        throw std::runtime_error("Attempted to receive " + std::to_string(count) + " list points but max was " + std::to_string(maxCount));

    if(count * m_config.model.D > dst.capacity())
        dst.reserve(count * m_config.model.D);

    if(count > dst.capacity())
        throw std::runtime_error("Failed to allocate memory for " + std::to_string(count) + " list points");

    MMPI_Recv(dst.data(), count * m_config.model.D, DATA_MPI_TYPE, mpiSource, TAG_RESULTS_DATA, MPI_COMM_WORLD, &status);
    return count;
}

void DesFramework::syncWithSlaves() const {
    // MPI_Barrier(MPI_COMM_WORLD);
    int a = 0;
    MPI_Bcast(&a, 1, MPI_INT, 0, MPI_COMM_WORLD);
}
