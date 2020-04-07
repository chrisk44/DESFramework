#pragma once

#include <cmath>
#include <iostream>

#include "framework.h"

using namespace std;

ParallelFramework::ParallelFramework(Limit* limits, ParallelFrameworkParameters& parameters) {
	unsigned int i;
	valid = false;

	// TODO: Verify parameters
	if (parameters.D == 0 || parameters.D>MAX_DIMENSIONS) {
		cout << "[E] Dimension must be between 0 and " << MAX_DIMENSIONS << endl;;
		return;
	}

	for (i = 0; i < parameters.D; i++) {
		if (limits[i].lowerLimit > limits[i].upperLimit) {
			cout << "[E] Limits for dimension " << i << ": Lower limit can't be higher than upper limit" << endl;
			return;
		}

		if (limits[i].N == 0) {
			cout << "[E] Limits for dimension " << i << ": N must be > 0" << endl;
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
	totalElements = (long)idxSteps[parameters.D - 1] * limits[parameters.D - 1].N;
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

int ParallelFramework::masterThread(MPI_Comm& comm, int numOfProcesses) {
	int finished = 0;

	//MPI_Comm_size(comm, &numOfProcesses);

	MPI_Status status;
	int mpiSource;
	long* startingIndices = new long[100];				// TODO: numOfProcesses might change, this should be allocated dynamically (numOfProcesses might also not be valid)

	bool* tmpResults = new bool[parameters->batchSize];
	long* tmpToCalculate = new long[parameters->D];
	int tmpNumOfElements;

	printf("\nstartingIndices: %x\n", startingIndices);
	printf("tmpResults: %x\n", tmpResults);
	printf("tmpToCalculate: %x\n", tmpToCalculate);
	printf("&numOfProcesses: %x\n", &numOfProcesses);
	printf("numOfProcesses: %d\n", numOfProcesses);
	printf("&tmpNumOfElements: %x\n", &tmpNumOfElements);
	printf("idxSteps: %x\n", idxSteps);
	printf("steps: %x\n", steps);
	printf("results: %x\n", results);
	printf("toSendVector: %x\n\n", toSendVector);

	while (finished < numOfProcesses) {
		// Receive request from any worker thread
		MPI_Recv(tmpResults, parameters->batchSize, MPI_CXX_BOOL, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
		cout << "[] Master: Received " << status.MPI_TAG << " from " << status.MPI_SOURCE << endl;
		mpiSource = status.MPI_SOURCE;

		if (status.MPI_TAG == TAG_READY) {
			// Get next data batch to calculate
			getDataChunk(tmpToCalculate, &tmpNumOfElements);
			startingIndices[mpiSource] = getIndexFromIndices(tmpToCalculate);

			// Send data
			cout << " Master: Sending " << tmpNumOfElements << " elements to " << mpiSource << " with index " << startingIndices[mpiSource] << endl;
			MPI_Send(&tmpNumOfElements, 1, MPI_INT, mpiSource, TAG_DATA_COUNT, comm);
			MPI_Send(tmpToCalculate, parameters->D, MPI_LONG, mpiSource, TAG_DATA, comm);

			// If no more data available, source will finish
			if (tmpNumOfElements == 0) {
				finished++;
				startingIndices[mpiSource] = -1;
			}

		}else if (status.MPI_TAG == TAG_RESULTS) {
			// Save received results in this->results
			MPI_Get_count(&status, MPI_CHAR, &tmpNumOfElements);
			memcpy(&results[startingIndices[mpiSource]], tmpResults, tmpNumOfElements);
		}

		// Update numOfProcesses, in case someone else joined in (TODO: is this even possible?)
		//MPI_Comm_size(comm, &numOfProcesses);
	}

	delete[] tmpResults;
	delete[] tmpToCalculate;
	delete[] startingIndices;

	return 0;
}

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
void ParallelFramework::getIndicesFromPoint(float* point, long* dst) {
	unsigned int i;

	for (i = 0; i < parameters->D; i++) {
		if (point[i] < limits[i].lowerLimit || point[i] >= limits[i].upperLimit) {
			cout << "Result query for out-of-bounds point" << endl;
			return;
		}

		// Calculate the steps for dimension i
		dst[i] = (int)floor(abs(limits[i].lowerLimit - point[i]) / steps[i]);
	}

#ifdef DEBUG
	cout << "Index for point ( ";
	for (i = 0; i < parameters->D; i++)
		cout << point[i] << " ";
	cout << "): ";

	for (i = 0; i < parameters->D; i++) {
		cout << dst[i] << " ";
	}
	cout << endl;
#endif
}
long ParallelFramework::getIndexFromIndices(long* pointIdx) {
	unsigned int i;
	long index = 0;

	for (i = 0; i < parameters->D; i++) {
		// Increase index by i*(index-steps for this dimension)
		index += pointIdx[i] * idxSteps[i];
	}

#ifdef DEBUG
	cout << "Index for point ( ";
	for (i = 0; i < parameters->D; i++)
		cout << pointIdx[i] << " ";
	cout << "): " << index << endl;
#endif

	return index;
}
