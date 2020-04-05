#ifndef PARALLELFRAMEWORK_H
#define PARALLELFRAMEWORK_H

#define cce() {                               \
  cudaError_t e = cudaGetLastError();                    \
  if (e != cudaSuccess) {                                \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                      \
    exit(1);                                             \
  }                                                      \
}

#define BLOCK_SIZE 1024
#define min(X, Y)  ((X) < (Y) ? (X) : (Y))
#define max(X, Y)  ((X) > (Y) ? (X) : (Y))

#include "cuda_runtime.h"
#include <cuda.h>
#include <iostream>

#define MAX_DIMENSIONS 10
//#define DEBUG

const int TYPE_CPU = 1;
const int TYPE_GPU = 2;

using namespace std;

class Model {
public:
	// float* point will be a D-dimension vector
	__host__   virtual bool validate_cpu(float* point) = 0;
	__device__ virtual bool validate_gpu(float point[]) = 0;

	virtual bool toBool() = 0;
};

struct Limit {
	float lowerLimit;
	float upperLimit;
	unsigned long N;
};

struct ParallelFrameworkParameters {
	unsigned int D;
	unsigned int computeBatchSize;
	unsigned int batchSize;
	// ...
};

// CUDA kernel to create the 'Model' object on device
template<class ImplementedModel>
__global__ void create_model_kernel(ImplementedModel** deviceModelAddress) {
	(*deviceModelAddress) = new ImplementedModel();
}

// CUDA kernel to delete the 'Model' object on device
template<class ImplementedModel>
__global__ void delete_model_kernel(ImplementedModel** deviceModelAddress) {
	delete (*deviceModelAddress);
}

// CUDA kernel to run the computation
template<class ImplementedModel>
__global__ void validate_kernel(ImplementedModel** model, long* startingPointIdx, bool* results, Limit* limits, unsigned int D) {
	float point[MAX_DIMENSIONS];
	long myIndex[MAX_DIMENSIONS];

	// Calculate myIndex = startingPointIdx + threadIdx.x
	unsigned int i;
	unsigned int carry = threadIdx.x;
	for (i = 0; i < D; i++) {
		myIndex[i] = (startingPointIdx[i] + carry) % limits[i].N;
		carry = (startingPointIdx[i] + carry) / limits[i].N;
	}

	// Calculate the exact point
	for (i = 0; i < D; i++) {
		point[i] = limits[i].lowerLimit + myIndex[i] * abs(limits[i].lowerLimit - limits[i].upperLimit) / limits[i].N;
	}

	// Run the validation function
	bool result = (*model)->validate_gpu(point);

	// Save the result to global memory
	results[threadIdx.x] = result;
}

class ParallelFramework {
private:
	// Parameters
	Limit* limits = NULL;			// This must be an array of length = parameters.D
	ParallelFrameworkParameters* parameters = NULL;

	// Runtime variables
	unsigned long* idxSteps = NULL;			// Index steps for each dimension
	float* steps = NULL;					// Real step for each dimension
	bool* results = NULL;					// An array of N0 * N1 * ... * ND
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
	int type = TYPE_GPU;

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

	// Model to use if type==TYPE_CPU
	ImplementedModel model = ImplementedModel();

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

				//model.validate_cpu();

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
		cudaFree(deviceLimits);			cce();
	}

	delete[] startPointIdx;
	delete[] tmpResults;

	return 0;
}

#endif