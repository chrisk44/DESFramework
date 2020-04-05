#ifndef PARALLELFRAMEWORK_H
#define PARALLELFRAMEWORK_H

#include "cuda_runtime.h"
#include <cuda.h>
#include <iostream>

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
__global__ void validate_kernel(ImplementedModel** model, float* points, bool* results) {
	int i = threadIdx.x;

	float point = points[i];

	bool result = (*model)->validate_gpu(&point);
	results[i] = result;
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
	long getIndexForPoint(float* point);
	bool isValid();

public:
	int masterThread();

	template<class ImplementedModel>
	int slaveThread(int type);

	void scanDimension(int d, float* prevDims, bool* results, int startIdx);
	void getDataChunk(long* toCalculate, int *numOfElements);

};

template<class ImplementedModel>
int ParallelFramework::run() {
	slaveThread<ImplementedModel>(0);

	int type;

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
	float* startPoint = new float[parameters->D];
	int numOfElements;

	// The device address where the device address of the model is saved
	ImplementedModel** deviceModelAddress;

	// Initialize GPU (instantiate an ImplementedModel object on the device)
	if (type == TYPE_GPU) {
		// Allocate space for the model's address on the device
		cudaMalloc(&deviceModelAddress, sizeof(ImplementedModel**));

		// Create the model object on the device, and write its address in 'deviceModelAddress' on the device
		create_model_kernel << < 1, 1 >> > (deviceModelAddress);
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
				//validate_kernel()
			}
			else if (type == TYPE_CPU) {

			}

			// Send the results to master

		}
		else {
			// No more data
			cout << "End of data" << endl;
			break;
		}
	}

	// Finalize GPU
	if (type == TYPE_GPU) {
		// Delete the model object on the device
		delete_model_kernel << < 1, 1 >> > (deviceModelAddress);

		// Free the space for the model's address on the device
		cudaFree(&deviceModelAddress);
	}

	delete[] startPointIdx;
	delete[] startPoint;

	return 0;
}

#endif