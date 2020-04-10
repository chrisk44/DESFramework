#pragma once

#include <cmath>
#include <iostream>
#include <Windows.h>

#include "framework.h"

using namespace std;

ParallelFramework::ParallelFramework(Limit* limits, ParallelFrameworkParameters& parameters) {
	unsigned int i;
	valid = false;

	// Verify parameters
	if (parameters.D == 0 || parameters.D>MAX_DIMENSIONS) {
		cout << "[E] Dimension must be between 1 and " << MAX_DIMENSIONS << endl;
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

	steps = new DATA_TYPE[parameters.D];
	for (i = 0; i < parameters.D; i++) {
		steps[i] = abs(limits[i].upperLimit - limits[i].lowerLimit) / limits[i].N;
	}

	totalSent = 0;
	totalElements = (long)idxSteps[parameters.D - 1] * limits[parameters.D - 1].N;
	results = new RESULT_TYPE[totalElements];		// Uninitialized
	// TODO: ^^ This really is a long story (memorywise)

	toSendVector = new unsigned long[parameters.D];
	for (i = 0; i < parameters.D; i++) {
		toSendVector[i] = 0;
	}

	this->limits = limits;
	this->parameters = &parameters;

	if (this->parameters->batchSize == 0) {
		// TODO: This really is a long story (memorywise)
		this->parameters->batchSize = totalElements;
	}

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
	SYSTEMTIME st;

	//MPI_Comm_size(comm, &numOfProcesses);

	MPI_Status status;
	int mpiSource;
	ComputeProcessDetails* pDetails = new ComputeProcessDetails[100];	// TODO: numOfProcesses might change, this should be allocated dynamically (numOfProcesses might also not be valid)

	RESULT_TYPE* tmpResults = new RESULT_TYPE[parameters->batchSize];	// TODO: batchSize might become larger, need to allocate more memory
	unsigned long* tmpToCalculate = new unsigned long[parameters->D];
	int tmpNumOfElements;	// This needs to be int because of MPI

#if DEBUG > 2
	printf("\npDetails: 0x%x\n", (void*) pDetails);
	printf("tmpResults: 0x%x\n", (void*) tmpResults);
	printf("tmpToCalculate: 0x%x\n", (void*) tmpToCalculate);
	printf("&numOfProcesses: 0x%x\n", (void*) &numOfProcesses);
	printf("numOfProcesses: %d\n", numOfProcesses);
	printf("&tmpNumOfElements: 0x%x\n", &tmpNumOfElements);
	printf("idxSteps: 0x%x\n", (void*)idxSteps);
	printf("steps: 0x%x\n", (void*) steps);
	printf("results: 0x%x\n", (void*) results);
	printf("toSendVector: 0x%x\n\n", (void*)toSendVector);
#endif

	while (finished < numOfProcesses) {
		// Receive request from any worker thread
		MPI_Recv(tmpResults, parameters->batchSize, RESULT_MPI_TYPE, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);		// TODO: batchSize is variable
		mpiSource = status.MPI_SOURCE;
#if DEBUG >= 2
		cout << " Master: Received " << status.MPI_TAG << " from " << status.MPI_SOURCE << endl;
#endif

		// Initialize process details if not initialized
		if (! (pDetails[mpiSource].initialized)) {
			// TODO: Add any more initializations
			pDetails[mpiSource].currentBatchSize = parameters->batchSize;
			pDetails[mpiSource].initialized = true;
		}

		if (status.MPI_TAG == TAG_READY) {
			// Get next data batch to calculate
			getDataChunk(pDetails[mpiSource].currentBatchSize, tmpToCalculate, &tmpNumOfElements);
			pDetails[mpiSource].computingIndex = getIndexFromIndices(tmpToCalculate);

			// Send data
#if DEBUG >= 2
			cout << " Master: Sending " << tmpNumOfElements << " elements to " << mpiSource << " with index " << pDetails[mpiSource].computingIndex << endl;
#endif
#if DEBUG >= 3
			cout << " Master: Sending data to " << mpiSource << ": ";
			for (unsigned int i = 0; i < parameters->D; i++) {
				cout << tmpToCalculate[i] << " ";
			}
			cout << endl;
#endif
			MPI_Send(&tmpNumOfElements, 1, MPI_INT, mpiSource, TAG_DATA_COUNT, comm);
			MPI_Send(tmpToCalculate, parameters->D, MPI_UNSIGNED_LONG, mpiSource, TAG_DATA, comm);

			// Update details for process
			GetSystemTime(&st);
			pDetails[mpiSource].computeStartTime = st.wMilliseconds;

			// If no more data available, source will finish
			if (tmpNumOfElements == 0) {
#if DEBUG >= 2
				cout << " Master: Slave " << mpiSource << " finishing..." << endl;
#endif
				finished++;
				pDetails[mpiSource].computingIndex = totalElements;
				pDetails[mpiSource].finished = true;
			}

		}else if (status.MPI_TAG == TAG_RESULTS) {
			// Save received results in this->results
			MPI_Get_count(&status, RESULT_MPI_TYPE, &tmpNumOfElements);
#if DEBUG >= 2
			printf(" Master: Saving %ld results from slave %d to results[%ld]...\n", tmpNumOfElements, mpiSource, pDetails[mpiSource].computingIndex);
#endif
#if DEBUG >= 4
			printf(" Master: Saving tmpResults: ");
			for (int i = 0; i < tmpNumOfElements; i++) {
				//printf("%d", min(tmpResults[i], 1));
				printf("%f ", tmpResults[i]);
			}
			printf(" at %d\n", pDetails[mpiSource].computingIndex);
#endif

			// Update details for process
			pDetails[mpiSource].jobsCompleted++;
			pDetails[mpiSource].elementsCalculated += tmpNumOfElements;

			GetSystemTime(&st);
			time_t completionTime = st.wMilliseconds - pDetails[mpiSource].computeStartTime;
			time_t newTimePerElement = completionTime / tmpNumOfElements;
			if (parameters->dynamicBatchSize) {
				// TODO: Adjust pDetails[mpiSource].currentBatchSize
			}
			pDetails[mpiSource].lastTimePerElement = newTimePerElement;
			memcpy(&results[pDetails[mpiSource].computingIndex], tmpResults, tmpNumOfElements*sizeof(RESULT_TYPE));


//#if DEBUG >= 4
//			printf(" Master: results after memcpy: ");
//			for (int i = 0; i < totalElements; i++) {
//				printf("%f ", results[i]);
//			}
//			printf("\n");
//#endif
		}

		// Update numOfProcesses, in case someone else joined in (TODO: is this even possible?)
		//MPI_Comm_size(comm, &numOfProcesses);
	}

	delete[] tmpResults;
	delete[] tmpToCalculate;
	delete[] pDetails;

	return 0;
}

void ParallelFramework::getDataChunk(unsigned long maxBatchSize, unsigned long* toCalculate, int* numOfElements) {
	if (totalSent >= totalElements) {
		*numOfElements = 0;
		return;
	}

	unsigned int adjustedBatchSize = maxBatchSize;
	if (totalElements - totalSent < adjustedBatchSize)
		adjustedBatchSize = totalElements - totalSent;

	// Copy toSendVector to the output
	memcpy(toCalculate, toSendVector, parameters->D * sizeof(long));
	*numOfElements = adjustedBatchSize;

	unsigned int i;
	unsigned int newIndex;
	unsigned int carry = adjustedBatchSize;

	for (i = 0; i < parameters->D; i++) {
		newIndex = (toSendVector[i] + carry) % limits[i].N;
		carry = (toSendVector[i] + carry) / limits[i].N;

		toSendVector[i] = newIndex;
	}

	totalSent += adjustedBatchSize;
}

RESULT_TYPE* ParallelFramework::getResults() {
	return results;
}
void ParallelFramework::getIndicesFromPoint(DATA_TYPE* point, unsigned long* dst) {
	unsigned int i;

	for (i = 0; i < parameters->D; i++) {
		if (point[i] < limits[i].lowerLimit || point[i] >= limits[i].upperLimit) {
			cout << "Result query for out-of-bounds point" << endl;
			return;
		}

		// Calculate the steps for dimension i
		dst[i] = (int) round(abs(limits[i].lowerLimit - point[i]) / steps[i]);		// TODO: 1.9999997 will round to 2, verify correctness
	}

//#if DEBUG >= 4
//	cout << "Index for point ( ";
//	for (i = 0; i < parameters->D; i++)
//		cout << point[i] << " ";
//	cout << "): ";
//
//	for (i = 0; i < parameters->D; i++) {
//		cout << dst[i] << " ";
//	}
//	cout << endl;
//#endif
}
long ParallelFramework::getIndexFromIndices(unsigned long* pointIdx) {
	unsigned int i;
	long index = 0;

	for (i = 0; i < parameters->D; i++) {
		// Increase index by i*(index-steps for this dimension)
		index += pointIdx[i] * idxSteps[i];
	}

//#if DEBUG >= 4
//	cout << "Index for point ( ";
//	for (i = 0; i < parameters->D; i++)
//		cout << pointIdx[i] << " ";
//	cout << "): " << index << endl;
//#endif

	return index;
}
