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
#include "definitions.h"

namespace desf {

class DesFramework {
private:
    // Parameters
    DesConfig m_config;

	// Runtime variables
    std::vector<unsigned long long> m_idxSteps;   // Index steps for each dimension
    int m_saveFile;							// File descriptor for save file
    char* m_finalResults;                   // An array of totalElements * sizeof(RESULT_TYPE) (this can't be a vector because we may want to map it to a file
    std::vector<std::vector<DATA_TYPE>> m_listResults; // A list of points for which the validation function has returned non-zero value
    unsigned long long m_totalReceived;		// Total elements that have been calculated and returned
    unsigned long long m_totalElements;		// Total elements

	// MPI
    int m_rank;

public:
    DesFramework(const DesConfig& config);
    ~DesFramework();

    void run();

    const decltype(m_config)& getConfig() const { return m_config; }
    const decltype(m_idxSteps)& getIndexSteps() const { return m_idxSteps; }

    const RESULT_TYPE* getResults() const;
    const std::vector<std::vector<DATA_TYPE>>& getList() const;

    void getIndicesFromPoint(DATA_TYPE* point, unsigned long* dst) const;
    unsigned long getIndexFromIndices(unsigned long* pointIdx) const;
    unsigned long getIndexFromPoint(DATA_TYPE* point) const;
    int getRank() const;

    template<typename T, T symbol>
    static T getGpuPointerFromSymbol(int gpuId) {
        cudaSetDevice(gpuId);

        T result;

        /*
         * This is the suggested way: Use cudaMemcpyFromSymbol(dst, symbol, size) to retrieve the address of the symbol.
         * However, it fails at runtime with a "invalid device symbol" error. No idea why.
         */
//        cudaMemcpyFromSymbol(&result, symbol, sizeof(T)); cce();

        /*
         * This is more complicated to understand but works. We call a kernel that obviously runs on the device, which
         * will save the address of the symbol to the device's memory which we allocate. Then we memcpy that memory back to the host.
         * The retrieved address makes absolutely no sense on the host, but can be passed to a __device__ or __global__ function
         * and will work correctly.
         */
        // Allocate memory on the device
        T* deviceMem = nullptr;
        cudaMalloc(&deviceMem, sizeof(T)); cce();

        // Run the kernel that saves the address of the symbol to the allocated memory
        getPointerFromSymbol<T, symbol><<<1, 1>>>(deviceMem);
        cudaDeviceSynchronize(); cce();

        // Retrieve the saved address from the allocated memory
        cudaMemcpy(&result, deviceMem, sizeof(T), cudaMemcpyDeviceToHost); cce();

        // Release the memory
        cudaFree(deviceMem); cce();

        return result;
    }

    template<typename T, T symbol>
    static std::map<int, T> getGpuPointersFromSymbol() {
        int numOfDevices;
        cudaGetDeviceCount(&numOfDevices);

        std::map<int, T> pointers;
        for(int i=0; i<numOfDevices; i++){
            pointers[i] = getGpuPointerFromSymbol<T, symbol>(i);
        }

        return pointers;
    }

    static int getNumOfProcesses();
    static int receiveRequest(int& source);
    static void sendBatch(const AssignedWork& work, int mpiSource);

    static void receiveAllResults(RESULT_TYPE* dst, size_t count, int mpiSource);
    static int receiveListResults(std::vector<size_t>& dst, size_t maxCount, int mpiSource);
    static void sync();

    static std::map<int, unsigned long> receiveMaxBatchSizes();
    static void sendMaxBatchSize(size_t maxBatchSize);

    static void sendReadyRequest();
    static AssignedWork receiveWorkFromMaster();
    static void sendResults(RESULT_TYPE* data, size_t count);
    static void sendListResults(size_t* data, size_t numOfPoints);
    static void sendExitSignal();

private:
    void masterProcess();
    void getPointFromIndex(unsigned long index, DATA_TYPE* result) const;

    void slaveProcess();

    void log(const char* text, ...);
};

}
