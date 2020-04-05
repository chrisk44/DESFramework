#pragma once

#include <cuda.h>
#include <cmath>
#include <iostream>

#include "framework.h"
#include "device_launch_parameters.h"

//#define DEBUG

using namespace std;

ParallelFramework::ParallelFramework(Limit* limits, ParallelFrameworkParameters& parameters) {
	unsigned int i;
	valid = false;

	// TODO: Verify parameters
	if (parameters.D == 0) {
		cout << "[E] Dimension must be > 0";
		return;
	}

	for (i = 0; i < parameters.D; i++) {
		if (limits[i].lowerLimit > limits[i].upperLimit) {
			cout << "[E] Limits for dimension " << i << ": Lower limit can't be higher than upper limit";
			return;
		}

		if (limits[i].N == 0) {
			cout << "[E] Limits for dimension " << i << ": N must be > 0";
			return;
		}
	}

	idxSteps = new unsigned long[parameters.D];
	idxSteps[0] = 1;
	for (i = 1; i < parameters.D; i++) {
		idxSteps[i] = idxSteps[i - 1] * limits[i-1].N;
	}

	steps = new float[parameters.D];
	for (i = 0; i < parameters.D; i++) {
		steps[i] = abs(limits[i].upperLimit - limits[i].lowerLimit) / limits[i].N;
	}

	totalSent = 0;
	totalElements = (long)idxSteps[parameters.D - 1];
	results = new bool[totalElements];		// Uninitialized

	toSendVector = new unsigned long[parameters.D];
	for (i = 0; i < parameters.D; i++) {
		toSendVector[i] = 0;
	}

	this->limits = limits;
	this->parameters = &parameters;

	valid = true;
}

ParallelFramework::~ParallelFramework() {
	delete [] idxSteps;
	delete [] steps;
	delete [] results;
	delete [] toSendVector;
	valid = false;
}

bool ParallelFramework::isValid() {
	return valid;
}
/*
template<class ImplementedModel>
int ParallelFramework::run<ImplementedModel>() {
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
	* /
	// Finalize MPI

	return 0;
}
*/
int ParallelFramework::masterThread() {

	// While (finished_processes < total_processes)
	while (true) {
		// Receive from any slave
		// If 'ready'
			// getDataChunk
			// If more data available
				// send data
			// else
				// notify about finish
				// finished_processes++

		// else if 'results_ready'
			// save received results in this->results
	}


	return 0;
}
/*
template<class ImplementedModel>
int ParallelFramework::slaveThread(int type) {
	long* startPointIdx = new long[parameters->D];
	float* startPoint = new float[parameters->D];
	int numOfElements;

	// The device address where the device address of the model is saved
	ImplementedModel** deviceModelAddress;

	if (type == TYPE_GPU) {
		// Allocate space for the model's address on the device
		cudaMalloc(&deviceModelAddress, sizeof(ImplementedModel**));

		// Create the model object on the device, and write its address in 'deviceModelAddress' on the device
		create_model_kernel<<< 1, 1 >>> (deviceModelAddress);
	}

	while (true) {
		// Send 'ready' signal to master

		// Receive data to compute
		getDataChunk(startPointIdx, &numOfElements);

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
			}else if (type == TYPE_CPU) {

			}

			// Send the results to master

		}else {
			// No more data
			cout << "End of data" << endl;
			break;
		}
	}

	delete [] startPointIdx;
	delete [] startPoint;

	if (type == TYPE_GPU) {
		// Allocate space for the model's address on the device
		cudaFree(&deviceModelAddress);

		// Create the model object on the device, and write its address in 'deviceModelAddress' on the device
		delete_model_kernel << < 1, 1 >> > (deviceModelAddress);
	}

	return 0;
}
*/
void ParallelFramework::getDataChunk(long* toCalculate, int* numOfElements) {
	if (totalSent >= totalElements) {
		*numOfElements = 0;
		return;
	}

	unsigned int adjustedBatchSize = parameters->batchSize;
	if (totalElements - totalSent < adjustedBatchSize)
		adjustedBatchSize = totalElements - totalSent;

	// Copy toSendVector to the output
	memcpy(toCalculate, toSendVector, parameters->D * sizeof(long));
	*numOfElements = adjustedBatchSize;

	unsigned int i;
	unsigned int newIndex;
	unsigned int carry = parameters->batchSize;

	for (i = 0; i < parameters->D; i++) {
		newIndex = (toSendVector[i] + carry) % limits[i].N;
		carry = (toSendVector[i] + carry) / limits[i].N;

		toSendVector[i] = newIndex;
	}

	totalSent += adjustedBatchSize;
}

bool* ParallelFramework::getResults() {
	return results;
}
long ParallelFramework::getIndexForPoint(float* point) {
	unsigned int i;
	long index = 0;
	int dimSteps;

	for (i = 0; i < parameters->D; i++) {
		if (point[i] < limits[i].lowerLimit || point[i] >= limits[i].upperLimit) {
			cout << "Result query for out-of-bounds point" << endl;
			return false;
		}

		// Calculate the steps for dimension i
		dimSteps = (int) floor(abs(limits[i].lowerLimit - point[i]) / steps[i] );

		// Increase index by i*(index-steps for this dimension)
		index += dimSteps * idxSteps[i];
	}

#ifdef DEBUG
	cout << "Index for point ( ";
	for (int j = 0; j < parameters->D; j++)
		cout << point[j] << " ";
	cout << "): " << index << endl;
#endif

	return index;
}
