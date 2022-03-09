#pragma once

#include <cuda.h>
#include <nvml.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <functional>

#include "utilities.h"
#include "cpuKernel.h"
#include "gpuKernel.h"

typedef std::function<void(ComputeThreadInfo&, ThreadCommonData*)> CallComputeThreadCallback;
typedef std::function<void(RESULT_TYPE*, Limit*, unsigned int, unsigned long, void*, int*, unsigned long long*, unsigned long, bool, int)> CallCpuKernelCallback;
typedef std::function<void(int, int, unsigned long, cudaStream_t, RESULT_TYPE*, unsigned long, const unsigned int, const unsigned long, const unsigned long, void*, int, bool, bool, int*, const int)> CallGpuKernelCallback;

class ParallelFramework {
private:
	// Parameters
    std::vector<Limit> limits;						// This must be an array of length = parameters.D
    ParallelFrameworkParameters parameters;

	// Runtime variables
    std::vector<unsigned long long> idxSteps;   // Index steps for each dimension
	int saveFile = -1;							// File descriptor for save file
    RESULT_TYPE* finalResults = NULL;			// An array of N0 * N1 * ... * N(D-1) (this can't be a vector because we may want to map it to a file
    std::vector<std::vector<DATA_TYPE>> listResults; // A list of points for which the validation function has returned non-zero value
	bool valid = false;
	unsigned long totalSent = 0;			// Total elements that have been sent for processing, also the index from which the next assigned batch will start
    unsigned long totalReceived = 0;		// Total elements that have been calculated and returned
	unsigned long totalElements = 0;		// Total elements

	// MPI
	int rank = -1;

public:
	ParallelFramework(bool initMPI);
	~ParallelFramework();
    void init(const std::vector<Limit>& limits, const ParallelFrameworkParameters& parameters);

	template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
    int run();

	RESULT_TYPE* getResults();
    std::vector<std::vector<DATA_TYPE>> getList();
	void getIndicesFromPoint(DATA_TYPE* point, unsigned long* dst);
	unsigned long getIndexFromIndices(unsigned long* pointIdx);
	unsigned long getIndexFromPoint(DATA_TYPE* point);
	bool isValid();
	int getRank();

private:
	void masterProcess();
    void coordinatorThread(ComputeThreadInfo* cti, ThreadCommonData* tcd, int numOfThreads);
	void getPointFromIndex(unsigned long index, DATA_TYPE* result);

    template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
    void slaveProcess();
    void slaveProcessImpl(CallComputeThreadCallback callComputeThread);

    template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
    void computeThread(ComputeThreadInfo& cti, ThreadCommonData* tcd);
    void computeThreadImpl(ComputeThreadInfo& cti, ThreadCommonData* tcd, CallCpuKernelCallback callCpuKernel, CallGpuKernelCallback callGpuKernel);

    int getNumOfProcesses() const;
    int receiveRequest(int& source) const;
    unsigned long receiveMaxBatchSize(int mpiSource) const;
    void sendBatchSize(const AssignedWork& work, int mpiSource) const;

    void receiveAllResults(RESULT_TYPE* dst, size_t count, int mpiSource) const;
    int receiveListResults(DATA_TYPE* dst, size_t maxCount, int mpiSource) const;
    void syncWithSlaves() const;
};

template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
int ParallelFramework::run() {
    if(!valid){
        std::string error = "[" + std::to_string(rank) + "] run() called for invalid framework";
        throw std::runtime_error(error);
    }

    if(rank == 0){
        if(parameters.printProgress) printf("[%d] Master process starting\n", rank);
        masterProcess();
        if(parameters.printProgress) printf("[%d] Master process finished\n", rank);
    }else{
        if(parameters.printProgress) printf("[%d] Slave process starting\n", rank);
        slaveProcess<validation_cpu, validation_gpu, toBool_cpu, toBool_gpu>();
        if(parameters.printProgress) printf("[%d] Slave process finished\n", rank);
    }

    if(parameters.finalizeAfterExecution)
        MPI_Finalize();

    return valid ? 0 : -1;
}

template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
void ParallelFramework::slaveProcess() {
    slaveProcessImpl([&](ComputeThreadInfo& cti, ThreadCommonData* tcd){
        computeThread<validation_cpu, validation_gpu, toBool_cpu, toBool_gpu>(cti, tcd);
    });
}

template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
void ParallelFramework::computeThread(ComputeThreadInfo& cti, ThreadCommonData* tcd) {
    CallCpuKernelCallback callCpuKernel = [&](RESULT_TYPE* results, Limit* limits, unsigned int D, unsigned long numOfElements, void* dataPtr, int* listIndexPtr,
            unsigned long long* idxSteps, unsigned long startingPointLinearIndex, bool dynamicScheduling, int batchSize){
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

    computeThreadImpl(cti, tcd, callCpuKernel, callGpuKernel);
}

