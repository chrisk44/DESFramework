#include <cmath>
#include <iostream>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <fcntl.h>

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
	if(! (parameters.benchmark))
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

void ParallelFramework::masterProcess() {
	MPI_Status status;
	int mpiSource;

	int pstatusAllocated = 0;
	ComputeProcessStatus* processStatus = nullptr;
	#define pstatus (processStatus[mpiSource])

	unsigned long allocatedElements = parameters->batchSize;				// Number of allocated elements for results
	RESULT_TYPE* tmpResults = new RESULT_TYPE[allocatedElements];
	unsigned long* tmpToCalculate = new unsigned long[parameters->D];
	int tmpNumOfElements;	// This needs to be int because of MPI
	float completionTime;

	while (totalReceived < totalElements) {
		// Receive request from any worker thread
		MPI_Recv(nullptr, 0, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		mpiSource = status.MPI_SOURCE;
		#if DEBUG >= 2
		cout << " Master: Received " << status.MPI_TAG << " from " << status.MPI_SOURCE << endl;
		#endif

		if(mpiSource+1 > pstatusAllocated){
			// Process joined in, allocate more memory
			processStatus = (ComputeProcessStatus*) realloc(processStatus, (mpiSource+1) * sizeof(ComputeProcessStatus));

			// Initialize the new ones
			for(int i=pstatusAllocated; i<mpiSource+1; i++){
				// TODO: Add any more initializations
				processStatus[i].currentBatchSize = parameters->batchSize;
				processStatus[i].computingIndex = 0;
			    processStatus[i].assignedElements = 0;
				processStatus[i].jobsCompleted = 0;
				processStatus[i].elementsCalculated = 0;
				processStatus[i].finished = false;

				pstatus.initialized = true;
			}

			pstatusAllocated = mpiSource+1;
		}

		switch(status.MPI_TAG){
			case TAG_READY:
				// Receive the maximum batch size reported by the slave process
				MPI_Recv(&pstatus.maxBatchSize, 1, MPI_UNSIGNED_LONG, mpiSource, TAG_MAX_DATA_COUNT, MPI_COMM_WORLD, &status);

				// Get next data batch to calculate
				getDataChunk(pstatus.currentBatchSize, tmpToCalculate, &tmpNumOfElements);
				pstatus.computingIndex = getIndexFromIndices(tmpToCalculate);

				#if DEBUG >= 2
				cout << " Master: Sending " << tmpNumOfElements << " elements to " << mpiSource << " with index " << pstatus.computingIndex << endl;
				#endif
				#if DEBUG >= 3
				cout << " Master: Sending data to " << mpiSource << ": ";
				for (unsigned int i = 0; i < parameters->D; i++) {
					cout << tmpToCalculate[i] << " ";
				}
				cout << endl;
				#endif

				// Send data
				MPI_Send(&tmpNumOfElements, 1, MPI_INT, mpiSource, TAG_DATA_COUNT, MPI_COMM_WORLD);
				MPI_Send(tmpToCalculate, parameters->D, MPI_UNSIGNED_LONG, mpiSource, TAG_DATA, MPI_COMM_WORLD);

				// Update details for process
				pstatus.stopwatch.start();
				pstatus.assignedElements = tmpNumOfElements;
				break;

			case TAG_RESULTS:
				// Receive the results
				MPI_Recv(tmpResults, pstatus.maxBatchSize, RESULT_MPI_TYPE, mpiSource, TAG_RESULTS_DATA, MPI_COMM_WORLD, &status);

				// Find the length of the results
				MPI_Get_count(&status, RESULT_MPI_TYPE, &tmpNumOfElements);	// This is equal to pstatus.assignedElements

				#if DEBUG >= 2
				printf(" Master: Saving %ld results from slave %d to results[%ld]...\n", tmpNumOfElements, mpiSource, pstatus.computingIndex);
				#endif
				#if DEBUG >= 4
				printf(" Master: Saving tmpResults: ");
				for (int i = 0; i < tmpNumOfElements; i++) {
					//printf("%d", min(tmpResults[i], 1));
					printf("%f ", tmpResults[i]);
				}
				printf(" at %d\n", pstatus.computingIndex);
				#endif

				this->totalReceived += tmpNumOfElements;

				// Update details for process
				pstatus.jobsCompleted++;
				pstatus.elementsCalculated += tmpNumOfElements;
				pstatus.stopwatch.stop();

				completionTime = pstatus.stopwatch.getMsec();

				if (parameters->benchmark) {
					printf("Slave %d: Benchmark: %d elements, %f ms\n", mpiSource, pstatus.assignedElements, completionTime);
					fflush(stdout);
				}

				if (parameters->dynamicBatchSize) {
					// Increase batch size until we hit the max

					// Adjust pstatus.currentBatchSize: Double until SS_THRESHOLD, then increse by SS_STEP
					if (pstatus.currentBatchSize < SS_THRESHOLD) {
						pstatus.currentBatchSize = std::min((int)(2*pstatus.currentBatchSize), (int)SS_THRESHOLD);
					} else {
						pstatus.currentBatchSize += SS_STEP;
					}

					// Make sure we haven't exceded the maximum batch size set by the process
					pstatus.currentBatchSize = min(pstatus.currentBatchSize, pstatus.maxBatchSize);

					if (allocatedElements < pstatus.currentBatchSize) {
						#if DEBUG >= 2
						printf("Master: Allocating more memory (%d -> %d elements, %ld MB)\n", allocatedElements, pstatus.currentBatchSize, pstatus.currentBatchSize*sizeof(RESULT_TYPE)/(1024*1024));
						#endif

						allocatedElements = pstatus.currentBatchSize;
						tmpResults = (RESULT_TYPE*)realloc(tmpResults, allocatedElements * sizeof(RESULT_TYPE));

						#if DEBUG >= 2
						printf("Master: tmpResults: 0x%x\n", (void*)tmpResults);
						#endif
					}
				}

				if( ! (parameters->benchmark))
					memcpy(&results[pstatus.computingIndex], tmpResults, tmpNumOfElements*sizeof(RESULT_TYPE));

				pstatus.assignedElements = 0;
				break;

			case TAG_EXITING:
				#if DEBUG >= 2
				cout << " Master: Slave " << mpiSource << " exiting..." << endl;
				#endif

				if(pstatus.assignedElements != 0){
					printf("[E] Slave %d exited with %d assigned elements!!\n", mpiSource, pstatus.assignedElements);
				}

				pstatus.computingIndex = totalElements;
				pstatus.finished = true;

				// TODO: Maybe set pstatus initialized = false to reuse the slot??
				break;
		}
	}

	delete[] tmpResults;
	delete[] tmpToCalculate;
	free(processStatus);
}

void ParallelFramework::coordinatorThread(ProcessingThreadInfo* pti, int numOfThreads){
	sem_t* semResults = pti[0].semResults;

	int numOfElements, elementsPerThread;
	unsigned long maxBatchSize = getDefaultCPUBatchSize();
	unsigned long *startPointIdx = new unsigned long[parameters->D];
	unsigned long allocatedElements = 0;
	RESULT_TYPE* results = nullptr;
	MPI_Status status;

	while(true){
		// Send READY signal to master
		MPI_Send(nullptr, 0, MPI_INT, 0, TAG_READY, MPI_COMM_WORLD);
		MPI_Send(&maxBatchSize, 1, MPI_UNSIGNED_LONG, 0, TAG_MAX_DATA_COUNT, MPI_COMM_WORLD);

		// Receive a batch of data from master
		MPI_Recv(&numOfElements, 1, MPI_INT, 0, TAG_DATA_COUNT, MPI_COMM_WORLD, &status);
		MPI_Recv(startPointIdx, parameters->D, MPI_UNSIGNED_LONG, 0, TAG_DATA, MPI_COMM_WORLD, &status);

		// If no results, break
		if(numOfElements == 0)
			break;

		// Make sure we have enough allocated memory for the results
		if(numOfElements > allocatedElements){
			allocatedElements = numOfElements;
			results = (RESULT_TYPE*) realloc(results, allocatedElements * sizeof(RESULT_TYPE));
			// TODO: assert results!=nullptr
		}

		// Split the data into numOfThreads pieces
		elementsPerThread = numOfElements / numOfThreads;

		pti[0].numOfElements = elementsPerThread>0 ? elementsPerThread : numOfElements;
		pti[0].results = results;

		for(int i=1; i<numOfThreads; i++){
			pti[i].numOfElements = elementsPerThread;
			pti[i].results = pti[i-1].results + pti[i-1].numOfElements;

			sem_post(&pti[i].semData);
		}

		// Wait for all threads to finish their work
		for(int i=0; i<numOfThreads; i++){
			sem_wait(semResults);
		}

		// Send all results to master
		MPI_Send(nullptr, 0, MPI_INT, 0, TAG_RESULTS, MPI_COMM_WORLD);
		MPI_Send(results, numOfElements, RESULT_MPI_TYPE, 0, TAG_RESULTS_DATA, MPI_COMM_WORLD);
	}

	// Notify about exiting
	MPI_Send(nullptr, 0, MPI_INT, 0, TAG_EXITING, MPI_COMM_WORLD);

	for(int i=0; i<numOfThreads; i++){
		pti[i].numOfElements = 0;
		sem_post(&pti[i].semData);
	}

	delete[] startPointIdx;
	free(results);
}

void ParallelFramework::getDataChunk(unsigned long batchSize, unsigned long* toCalculate, int* numOfElements) {
	if (totalSent >= totalElements) {
		*numOfElements = 0;
		return;
	}

	if (totalElements - totalSent < batchSize)
		batchSize = totalElements - totalSent;

	// Copy toSendVector to the output
	memcpy(toCalculate, toSendVector, parameters->D * sizeof(long));
	*numOfElements = batchSize;

	unsigned int i;
	unsigned int newIndex;
	unsigned int carry = batchSize;

	for (i = 0; i < parameters->D; i++) {
		newIndex = (toSendVector[i] + carry) % limits[i].N;
		carry = (toSendVector[i] + carry) / limits[i].N;

		toSendVector[i] = newIndex;
	}

	totalSent += batchSize;
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
