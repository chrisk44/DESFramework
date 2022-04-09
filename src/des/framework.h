#pragma once

#include <cuda.h>
#include <nvml.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <list>
#include <map>
#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <functional>

#include "computeThread.h"
#include "cpuKernel.h"
#include "gpuKernel.h"

typedef std::function<void(ComputeThread&, ThreadCommonData&)> ComputeThreadStarter;

class ParallelFramework {
private:
	// Parameters
    std::vector<Limit> m_limits;						// This must be an array of length = parameters.D
    ParallelFrameworkParameters m_parameters;

	// Runtime variables
    std::vector<unsigned long long> m_idxSteps;   // Index steps for each dimension
    int m_saveFile;							// File descriptor for save file
    RESULT_TYPE* m_finalResults;			// An array of N0 * N1 * ... * N(D-1) (this can't be a vector because we may want to map it to a file
    std::vector<std::vector<DATA_TYPE>> m_listResults; // A list of points for which the validation function has returned non-zero value
    bool m_valid;
    unsigned long m_totalSent;			// Total elements that have been sent for processing, also the index from which the next assigned batch will start
    unsigned long m_totalReceived;		// Total elements that have been calculated and returned
    unsigned long m_totalElements;		// Total elements

	// MPI
    int m_rank;

public:
    ParallelFramework(bool initMPI);
    ~ParallelFramework();
    void init(const std::vector<Limit>& limits, const ParallelFrameworkParameters& parameters);

	template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
    int run();

    const decltype(m_parameters)& getParameters() const { return m_parameters; }
    const decltype(m_limits)& getLimits() const { return m_limits; }
    const decltype(m_idxSteps)& getIndexSteps() const { return m_idxSteps; }

    const RESULT_TYPE* getResults() const;
    const std::vector<std::vector<DATA_TYPE>>& getList() const;

    void getIndicesFromPoint(DATA_TYPE* point, unsigned long* dst) const;
    unsigned long getIndexFromIndices(unsigned long* pointIdx) const;
    unsigned long getIndexFromPoint(DATA_TYPE* point) const;
    bool isValid() const;
    int getRank() const;

private:
    void masterProcess();
    void coordinatorThread(std::vector<ComputeThread>& cti, ThreadCommonData& tcd);
    void getPointFromIndex(unsigned long index, DATA_TYPE* result) const;

    template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
    void slaveProcess();
    void slaveProcessImpl(CallCpuKernelCallback callCpuKernel, CallGpuKernelCallback callGpuKernel);

    int getNumOfProcesses() const;
    int receiveRequest(int& source) const;
    unsigned long receiveMaxBatchSize(int mpiSource) const;
    void sendBatchSize(const AssignedWork& work, int mpiSource) const;

    void receiveAllResults(RESULT_TYPE* dst, size_t count, int mpiSource) const;
    int receiveListResults(DATA_TYPE* dst, size_t maxCount, int mpiSource) const;
    void syncWithSlaves() const;

    void sendReadyRequest(unsigned long maxBatchSize) const;
    AssignedWork receiveWorkFromMaster() const;
    void sendResults(RESULT_TYPE* data, size_t count) const;
    void sendListResults(DATA_TYPE* data, size_t numOfPoints) const;
    void sendExitSignal() const;
};

template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
int ParallelFramework::run() {
    if(!m_valid){
        std::string error = "[" + std::to_string(m_rank) + "] run() called for invalid framework";
        throw std::runtime_error(error);
    }

    if(m_rank == 0){
        if(m_parameters.printProgress) printf("[%d] Master process starting\n", m_rank);
        masterProcess();
        if(m_parameters.printProgress) printf("[%d] Master process finished\n", m_rank);
    }else{
        if(m_parameters.printProgress) printf("[%d] Slave process starting\n", m_rank);
        slaveProcess<validation_cpu, validation_gpu, toBool_cpu, toBool_gpu>();
        if(m_parameters.printProgress) printf("[%d] Slave process finished\n", m_rank);
    }

    if(m_parameters.finalizeAfterExecution)
        MPI_Finalize();

    return m_valid ? 0 : -1;
}

template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
void ParallelFramework::slaveProcess() {
    CallCpuKernelCallback callCpuKernel = [&](RESULT_TYPE* results, const Limit* limits, unsigned int D, unsigned long numOfElements, void* dataPtr, int* listIndexPtr,
            const unsigned long long* idxSteps, unsigned long startingPointLinearIndex, bool dynamicScheduling, int batchSize){
        cpu_kernel(validation_cpu, toBool_cpu, results, limits, D, numOfElements, dataPtr, listIndexPtr, idxSteps, startingPointLinearIndex, dynamicScheduling, batchSize);
    };

    CallGpuKernelCallback callGpuKernel = [&](int numOfBlocks, int blockSize, unsigned long sharedMem, cudaStream_t stream,
            RESULT_TYPE* results, unsigned long startingPointLinearIndex,
            const unsigned int D, const unsigned long numOfElements, const unsigned long offset, void* dataPtr,
            int dataSize, bool useSharedMemoryForData, bool useConstantMemoryForData, int* listIndexPtr,
            const int computeBatchSize){
        validate_kernel<validation_gpu, toBool_gpu><<<numOfBlocks, blockSize, sharedMem, stream>>>(
            results, startingPointLinearIndex, D, numOfElements, offset,
            dataPtr, dataSize, useSharedMemoryForData, useConstantMemoryForData,
            listIndexPtr, computeBatchSize
        );
    };

    slaveProcessImpl(callCpuKernel, callGpuKernel);
}

