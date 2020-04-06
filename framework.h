#ifndef PARALLELFRAMEWORK_H
#define PARALLELFRAMEWORK_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <mpi.h>

#include "utilities.h"
#include "kernels.cpp"

using namespace std;

struct ParallelFrameworkParameters {
	unsigned int D;
	unsigned int computeBatchSize;
	unsigned int batchSize;
	// ...
};

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
	int run();

	bool* getResults();
	void getIndicesFromPoint(float* point, long* dst);
	long getIndexFromIndices(long* pointIdx);
	bool isValid();

public:
	int masterThread();

	template<class ImplementedModel>
	int slaveThread(int type);

	void getDataChunk(long* toCalculate, int *numOfElements);

};

template<class ImplementedModel>
int ParallelFramework::run() {
	int type = TYPE_CPU;

	slaveThread<ImplementedModel>(type);


	// For each GPU, fork, set type = gpu

	// Fork once for cpu (initial thread must be the master), set type = cpu

	// Initialize MPI
	/*
	int rank = 0;

	if(rank == 0){
		masterThread();
	}else{
		slaveThread(type);
	}
	*/
	// Finalize MPI

	return 0;
}

template<class ImplementedModel>
int ParallelFramework::slaveThread(int type) {
	long* startPointIdx = new long[parameters->D];
	bool* tmpResults = new bool[parameters->batchSize];
	int numOfElements;
	int blockSize;
	int numOfBlocks;

	// The device address where the device address of the model is saved
	ImplementedModel** deviceModelAddress;
	bool* deviceResults;
	long* deviceStartingPointIdx;
	Limit* deviceLimits;

	// Initialize GPU (instantiate an ImplementedModel object on the device)
	if (type == TYPE_GPU) {
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

		// Receive data to compute
		// TODO: Receive from MPI
		getDataChunk(startPointIdx, &numOfElements);		// TODO: This will eventually be called by masterThread()

		// If received more data...
		if (numOfElements > 0) {
#ifdef DEBUG
			cout << "Got " << numOfElements << " elements: [";
			for (unsigned int i = 0; i < parameters->D; i++)
				cout << startPointIdx[i] << " ";
			cout << "]" << endl;
#endif

			// Calculate the results
			if (type == TYPE_GPU) {

				// Copy starting point indices to device
				cudaMemcpy(deviceStartingPointIdx, startPointIdx, parameters->D * sizeof(long), cudaMemcpyHostToDevice);
				cce();

				// Call the kernel
				blockSize = min(BLOCK_SIZE, numOfElements);
				numOfBlocks = (numOfElements + blockSize - 1) / blockSize;
				validate_kernel<ImplementedModel><<<numOfBlocks, blockSize>>>(deviceModelAddress, deviceStartingPointIdx, deviceResults, deviceLimits, parameters->D);
				cce();

				// Wait for kernel to finish
				cudaDeviceSynchronize();
				cce();

				// Get results from device
				cudaMemcpy(tmpResults, deviceResults, numOfElements * sizeof(bool), cudaMemcpyDeviceToHost);
				cce();

			}else if (type == TYPE_CPU) {

				cpu_kernel<ImplementedModel>(startPointIdx, tmpResults, limits, parameters->D, numOfElements);

			}

			// Send the results to master
			memcpy(&results[getIndexFromIndices(startPointIdx)], tmpResults, numOfElements * sizeof(bool));	// TODO: This will be done by masterThread

#ifdef DEBUG
			// Print results
			cout << "Results:";
			for (unsigned int i = 0; i < numOfElements; i++) {
				cout << " " << tmpResults[i];
			}
			cout << endl;
#endif
		}
		else {
			// No more data
			cout << "End of data" << endl << endl;
			break;
		}
	}

	// Finalize GPU
	if (type == TYPE_GPU) {
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