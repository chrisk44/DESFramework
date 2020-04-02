#pragma once

#include <cuda.h>
#include <cmath>
#include <iostream>

#include "model.cu"
#include "framework.h"
//#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#define DEBUG


using namespace std;

ParallelFramework::ParallelFramework(Limit* limits, ParallelFrameworkParameters& parameters, Model& model) {
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

	idxSteps = (unsigned long*)malloc(sizeof(long) * parameters.D);
	idxSteps[0] = limits[0].N;
	for (i = 1; i < parameters.D; i++) {
		idxSteps[i] = idxSteps[i - 1] * limits[i-1].N;
	}

	steps = (float*)malloc(sizeof(float) * parameters.D);
	for (i = 0; i < parameters.D; i++) {
		steps[i] = abs(limits[i].upperLimit - limits[i].lowerLimit) / limits[i].N;
	}

	long totalElements = (long)idxSteps[parameters.D - 1] * (long)limits[parameters.D - 1].N;
	results = (bool*)calloc(totalElements, sizeof(bool));

	toSendVector = (unsigned long*)calloc(parameters.D, sizeof(long));

	this->limits = limits;
	this->parameters = &parameters;
	this->model = &model;

	valid = true;
}

ParallelFramework::~ParallelFramework() {
	free(idxSteps);
	free(steps);
	free(results);
	free(toSendVector);
	valid = false;
}

bool ParallelFramework::isValid() {
	return valid;
}

int ParallelFramework::run() {

	int type;

	// For each GPU, fork, set type = gpu

	// Fork once for cpu (initial thread must be the master), set type = cpu

	// Initialize MPI

	int rank = 0;

	if(rank == 0){
		masterThread();
	}else{
		slaveThread(type);
	}

	// Finalize MPI

	return 0;
}

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
int ParallelFramework::slaveThread(int type) {

	while (true) {
		// Send 'ready' signal to master

		// Receive data to compute

		// If received more data...

			// Calculate the results

			// Send the results to master

		// No more data
			// break
	}

	return 0;
}

void ParallelFramework::getDataChunk(long *toCalculate, int *numOfElements) {
	// TODO: Eventually change it to assign parameters->batch_size elements, instead of limits[0].N

	// toSendVector[0] is initially 0
	// It becomes 1 when this function reaches the end of the rest of the dimensions
	if (toSendVector[0] != 0) {
		*numOfElements = 0;
		return;
	}

	// Copy toSendVector to the output
	memcpy(toCalculate, toSendVector, parameters->D * sizeof(long));

	unsigned int i;
	unsigned int carry = 1;
	for (i = 1; i < parameters->D; i++) {
		toSendVector += carry;

		if (toSendVector[i] == limits[i].N) {
			toSendVector[i] = 0;
			carry = 1;
		} else {
			carry = 0;
		}
	}

	if (carry != 0) {
		*numOfElements = 0;
		toSendVector[0] = 1;
	} else {
		*numOfElements = limits[0].N;
	}

	*numOfElements = carry == 1 ? 0 : limits[0].N;
}

/*
float* point = (float*)malloc(sizeof(float) * parameters->D);

// Start scanning from the last dimension
scanDimension(parameters->D - 1, point, results, 0);

free(point);
*/
/*
void ParallelFramework::scanDimension(int d, float* point, bool* results, int startIdx) {
	unsigned int i;
	// Recursively reach dimension 0, where the actual computation will be done

	if (d == 0) {
		// Go through every point in dimension 0, using the pre-set values in 'point' for the other dimensions
		// Calculate the result for point[:, const, const, ...]
		// Save in results[startIdx + 0:limits[d].N]

		// TODO: Dispatch this to GPU or CPU

#ifdef DEBUG
		cout << "Calculating with startIdx: " << startIdx << endl;
#endif

		for (i = 0; i < limits[d].N; i++) {
			// Set the point for this iteration
			point[d] = limits[d].lowerLimit + i * steps[d];

			// Get the result
			bool result = model->validate_cpu(point);
			
#ifdef DEBUG
			cout << "Result for point=";
			for (int j = 0; j < parameters->D; j++)
				cout << point[j] << " ";
			cout << "-> " << result << endl;
#endif

			// Save the result in 'results'
			results[startIdx + i] = result;
		}

	} else {
		// Recursive call for each point in dimension d

		// For each element in dimension d...
		for (i = 0; i < limits[d].N; i++) {
			// Set the point for this iteration
			point[d] = limits[d].lowerLimit + i * steps[d];

			// Adjust startIdx: i*N[d-1]*N[d-2]*...*N[0]
			int tmpStartIdx = startIdx + i*idxSteps[d-1];
			
#ifdef DEBUG
			cout << "Recursive call from D" << d << ", startIdx=" << startIdx << ", point=";
			for (int j = 0; j < parameters->D; j++)
				cout << point[j] << " ";
			cout << endl;
#endif

			scanDimension(d - 1, point, results, tmpStartIdx);
		}
	}
	
}
*/
bool* ParallelFramework::getResults() {
	return results;
}
bool ParallelFramework::getResultAt(float* point) {
	unsigned int i;
	int index = 0;
	int dimSteps;

	for (i = 0; i < parameters->D; i++) {
		if (point[i] < limits[i].lowerLimit || point[i] >= limits[i].upperLimit) {
			cout << "Result query for out-of-bounds point" << endl;
			return false;
		}

		// Calculate the steps for dimension i
		dimSteps = (int) floor(abs(limits[i].lowerLimit - point[i]) / steps[i] );

		// Increase index by i*(index-steps for previous dimension)
		index += dimSteps * (i > 0 ? idxSteps[i - 1] : 1);
	}

#ifdef DEBUG
	cout << "Index for point ( ";
	for (int j = 0; j < parameters->D; j++)
		cout << point[j] << " ";
	cout << "): " << index << endl;
#endif

	return results[index];
}

__global__ void validate_kernel(Model* model, float* points, bool* results) {
	int i = threadIdx.x;

	float point = points[i];

	bool result = model->validate_gpu(&point);
	results[i] = result;
}

