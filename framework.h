#ifndef PARALLELFRAMEWORK_H
#define PARALLELFRAMEWORK_H

#include <cuda.h>
#include <nvml.h>

#include <cmath>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <limits.h>
#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include "utilities.h"
#include "kernels.h"

using namespace std;

class ParallelFramework {
private:
	// Parameters
	Limit* limits = NULL;						// This must be an array of length = parameters.D
	ParallelFrameworkParameters* parameters = NULL;

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
	void init(Limit* limits, ParallelFrameworkParameters& parameters);

	template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
	int run();

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

template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
int ParallelFramework::run() {
	if(!valid){
		printf("[%d] run() called for invalid framework\n", rank);
		return -1;
	}

	if(rank == 0){

		if(parameters->printProgress)
			printf("[%d] Master process starting\n", rank);

		masterProcess();

		if(parameters->printProgress)
			printf("[%d] Master process finished\n", rank);

	}else{

		if(parameters->printProgress)
			printf("[%d] Slave process starting\n", rank);

		slaveProcess(validation_cpu, validation_gpu, toBool_cpu, toBool_gpu);

		if(parameters->printProgress)
			printf("[%d] Slave process finished\n", rank);

	}

	if(parameters->finalizeAfterExecution)
		MPI_Finalize();

	return valid ? 0 : -1;
}

#endif
