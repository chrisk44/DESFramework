#ifndef PARALLELFRAMEWORK_H
#define PARALLELFRAMEWORK_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <mpi.h>
#include <omp.h>

#include "utilities.h"
#include "kernels.cpp"

using namespace std;

class ParallelFramework {
private:
	// Parameters
	Limit* limits = NULL;			// This must be an array of length = parameters.D
	ParallelFrameworkParameters* parameters = NULL;

	// Runtime variables
	unsigned long* idxSteps = NULL;			// Index steps for each dimension
	float* steps = NULL;					// Real step for each dimension
	bool* results = NULL;					// An array of N0 * N1 * ... * N(D-1)
	bool valid = false;
	unsigned long* toSendVector = NULL;		// An array of D elements, where every entry shows the next element of that dimension to be dispatched
	unsigned long totalSent = 0;			// Total elements that have been sent for processing
	unsigned long totalElements = 0;		// Total elements

public:
	ParallelFramework(Limit* limits, ParallelFrameworkParameters& parameters);
	~ParallelFramework();

	template<class ImplementedModel>
	int run(char* argv0);

	bool* getResults();
	void getIndicesFromPoint(float* point, long* dst);
	long getIndexFromIndices(long* pointIdx);
	bool isValid();

public:
	int masterThread(MPI_Comm& comm, int numOfProcesses);

	template<class ImplementedModel>
	int slaveThread(int rank, MPI_Comm& comm);

	void getDataChunk(long maxBatchSize, long* toCalculate, int *numOfElements);

};

template<class ImplementedModel>
int ParallelFramework::run(char* argv0) {
	int numOfProcesses, tmp;
	char* programName;
	// MPI variables
	int rank;
	int* errcodes;
	MPI_Comm parentcomm, intercomm;

	// Initialize MPI
	MPI_Init(nullptr, nullptr);
	MPI_Comm_get_parent(&parentcomm);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Calculate number of processes (#GPUs + 1CPU)
	numOfProcesses = 0;
	if(! (parameters->cpuOnly))
		cudaGetDeviceCount(&numOfProcesses);
	numOfProcesses++;

	// Allocate errcodes now that numOfProcesses is known
	errcodes = new int[numOfProcesses];

	// Isolate the program's name from argv0
	tmp = (int)strlen(argv0) - 1;
	while (argv0[tmp] != '/' && argv0[tmp] != '\\') {
		tmp--;
	}
	programName = &argv0[tmp];

	// If this is the parent process, spawn children and run masterThread, else run slaveThread
	if (parentcomm == MPI_COMM_NULL) {
#if DEBUG >=1
		cout << "Master: Spawning " << numOfProcesses << " processes" << endl;
#endif

		MPI_Comm_spawn(programName, MPI_ARGV_NULL, numOfProcesses, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &intercomm, errcodes);

		// Check errorcodes
		for (int i = 0; i < numOfProcesses; i++) {
			if (errcodes[i] != MPI_SUCCESS) {
				cout << "[E] Error starting process " << i << ", error: " << errcodes[i] << endl;

				// TBD: Terminate everyone (?)
				//totalSent = totalElements;
			}
		}

		masterThread(intercomm, numOfProcesses);
#if DEBUG >=1
		cout << "Master finished" << endl;
#endif
	} else {
#if DEBUG >=1
		cout << "Slave " << rank << " starting" << endl;
#endif
		slaveThread<ImplementedModel>(rank, parentcomm);
#if DEBUG >=1
		cout << "Slave " << rank << " finished" << endl;
#endif

		MPI_Finalize();
		exit(0);
	}

	// Only parent (master) continues here
	MPI_Finalize();

	delete[] errcodes;

	return 0;
}

template<class ImplementedModel>
int ParallelFramework::slaveThread(int rank, MPI_Comm& comm) {
	// Rank 0 is cpu, 1+ are gpus

	int gpuId = rank - 1;

	long* startPointIdx = new long[parameters->D];
	bool* tmpResults = new bool[parameters->batchSize];
	int numOfElements;
	int blockSize;
	int numOfBlocks;
	MPI_Status status;

	// deviceModelAddress is where the device address of the model is saved
	ImplementedModel** deviceModelAddress;

	bool* deviceResults;
	long* deviceStartingPointIdx;
	Limit* deviceLimits;

	// Initialize GPU (instantiate an ImplementedModel object on the device)
	if (gpuId > -1) {
		// Select GPU with 'id'
		cudaSetDevice(gpuId);

		// Allocate memory on device
		cudaMalloc(&deviceModelAddress, sizeof(ImplementedModel**));		cce();
		cudaMalloc(&deviceResults, parameters->batchSize * sizeof(bool));	cce();
		cudaMalloc(&deviceStartingPointIdx, parameters->D * sizeof(long));	cce();
		cudaMalloc(&deviceLimits, parameters->D * sizeof(Limit));			cce();

		// Create the model object on the device, and write its address in 'deviceModelAddress' on the device
		create_model_kernel<ImplementedModel> << < 1, 1 >> > (deviceModelAddress);	cce();

		// Move limits to device
		cudaMemcpy(deviceLimits, limits, parameters->D * sizeof(Limit), cudaMemcpyHostToDevice);	cce();
	}

	while (true) {

		// Send 'ready' signal to master
#if DEBUG >= 2
		cout << "  Slave " << rank << " sending READY..." << endl;
#endif
		MPI_Send(nullptr, 0, MPI_INT, 0, TAG_READY, comm);

		// Receive data (length and starting point) to compute
#if DEBUG >= 2
		cout << "  Slave " << rank << " waiting for data..." << endl;
#endif
		MPI_Recv(&numOfElements, 1, MPI_LONG, 0, TAG_DATA_COUNT, comm, &status);
		MPI_Recv(startPointIdx, parameters->D, MPI_LONG, 0, TAG_DATA, comm, &status);

		// If received more data...
		if (numOfElements > 0) {
#if DEBUG >= 3
			cout << "Got " << numOfElements << " elements: ";
			for (unsigned int i = 0; i < parameters->D; i++)
				cout << startPointIdx[i] << " ";
			cout << endl;
#endif
				
			// Calculate the results
			if (gpuId > -1) {

				// Copy starting point indices to device
				cudaMemcpy(deviceStartingPointIdx, startPointIdx, parameters->D * sizeof(long), cudaMemcpyHostToDevice);
				cce();

				// Call the kernel
				blockSize = min(BLOCK_SIZE, numOfElements);
				numOfBlocks = (numOfElements + blockSize - 1) / blockSize;
				validate_kernel<ImplementedModel> << <numOfBlocks, blockSize >> > (deviceModelAddress, deviceStartingPointIdx, deviceResults, deviceLimits, parameters->D);
				cce();

				// Wait for kernel to finish
				cudaDeviceSynchronize();
				cce();

				// Get results from device
				cudaMemcpy(tmpResults, deviceResults, numOfElements * sizeof(bool), cudaMemcpyDeviceToHost);
				cce();

			} else {

				cpu_kernel<ImplementedModel>(startPointIdx, tmpResults, limits, parameters->D, numOfElements);

			}

			// Send the results to master
#if DEBUG >= 2
			cout << "  Slave " << rank << " sending RESULTS..." << endl;
#endif
			MPI_Send(tmpResults, numOfElements, MPI_CXX_BOOL, 0, TAG_RESULTS, comm);

#if DEBUG >= 4
			// Print results
			cout << "Results:";
			for (unsigned int i = 0; i < numOfElements; i++) {
				cout << " " << tmpResults[i];
			}
			cout << endl;
#endif
		} else {
			// No more data
#if DEBUG >= 2
			cout << "  Slave " << rank << " got 0 data..." << endl;
#endif
			break;
		}
	}

	// Finalize GPU
	if (gpuId > -1) {
		// Delete the model object on the device
		delete_model_kernel<ImplementedModel> << < 1, 1 >> > (deviceModelAddress);
		cce();

		// Free the space for the model's address on the device
		cudaFree(deviceModelAddress);		cce();
		cudaFree(deviceResults);			cce();
		cudaFree(deviceStartingPointIdx);	cce();
		cudaFree(deviceLimits);				cce();
	}

	delete[] startPointIdx;
	delete[] tmpResults;

	return 0;
}

#endif
