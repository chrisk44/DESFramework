#pragma once

#include <cuda.h>
#include <nvml.h>

#include <cmath>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

#include "utilities.h"
#include "kernels.h"

class ParallelFramework {
private:
	// Parameters
    std::vector<Limit> limits;						// This must be an array of length = parameters.D
    ParallelFrameworkParameters parameters;

	// Runtime variables
	unsigned long long* idxSteps = NULL;		// Index steps for each dimension
	int saveFile = -1;							// File descriptor for save file
	RESULT_TYPE* finalResults = NULL;			// An array of N0 * N1 * ... * N(D-1)
	DATA_TYPE* listResults = NULL;				// An array of points for which the validation function has returned non-zero value
	int listResultsSaved = 0;					// Number of points saved in listResults
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
    int run(){
        return _run(validation_cpu, validation_gpu, toBool_cpu, toBool_gpu);
    }

    int _run(validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu);

	RESULT_TYPE* getResults();
	DATA_TYPE* getList(int* length);
	void getIndicesFromPoint(DATA_TYPE* point, unsigned long* dst);
	unsigned long getIndexFromIndices(unsigned long* pointIdx);
	unsigned long getIndexFromPoint(DATA_TYPE* point);
	bool isValid();
	int getRank();

private:
	void masterProcess();
	void coordinatorThread(ComputeThreadInfo* cti, ThreadCommonData* tcd, int numOfThreads);
	void getPointFromIndex(unsigned long index, DATA_TYPE* result);

    void slaveProcess(validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu);
    void computeThread(ComputeThreadInfo& cti, ThreadCommonData* tcd, validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu);
};

