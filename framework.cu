#include <cmath>
#include <iostream>

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

	for (i = 0; i < parameters.D; i++) {
		limits[i].step = abs(limits[i].upperLimit - limits[i].lowerLimit) / limits[i].N;
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
	int finished = 0;
	int numOfProcesses;
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses);

	int pstatusAllocated = 0;
	ComputeProcessStatus* processStatus = nullptr;
	#define pstatus (processStatus[mpiSource])

	unsigned long allocatedElements = parameters->batchSize;				// Number of allocated elements for results
	RESULT_TYPE* tmpResults = new RESULT_TYPE[allocatedElements];
	unsigned long* tmpToCalculate = new unsigned long[parameters->D];
	int tmpNumOfElements;	// This needs to be int because of MPI
	float completionTime;


	while (totalReceived < totalElements || finished < numOfProcesses-1) {
		// Receive request from any worker thread
		#if DEBUG >= 2
			printf("Master: Waiting for signal...\n");
		#endif
		fflush(stdout);

		MMPI_Recv(nullptr, 0, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		mpiSource = status.MPI_SOURCE;

		#if DEBUG >= 2
			printf("Master: Received %d from %d\n", status.MPI_TAG, mpiSource);
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
				MMPI_Recv(&pstatus.maxBatchSize, 1, MPI_UNSIGNED_LONG, mpiSource, TAG_MAX_DATA_COUNT, MPI_COMM_WORLD, &status);

				// Get next data batch to calculate
				getDataChunk(pstatus.currentBatchSize, tmpToCalculate, &tmpNumOfElements);
				pstatus.computingIndex = getIndexFromIndices(tmpToCalculate);

				#if DEBUG >= 2
					printf("Master: Sending %d elements to %d with index %d\n", tmpNumOfElements, mpiSource, pstatus.computingIndex);
				#endif
				#if DEBUG >= 3
					printf("Master: Sending data to %d: ", mpiSource);
					for (unsigned int i = 0; i < parameters->D; i++) {
						printf("%d ", tmpToCalculate[i]);
					}
					printf("\n");
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
				MMPI_Recv(tmpResults, pstatus.maxBatchSize, RESULT_MPI_TYPE, mpiSource, TAG_RESULTS_DATA, MPI_COMM_WORLD, &status);

				// Find the length of the results
				MPI_Get_count(&status, RESULT_MPI_TYPE, &tmpNumOfElements);	// This is equal to pstatus.assignedElements

				#if DEBUG >= 2
					printf("Master: Saving %ld results from slave %d to results[%ld]...\n", tmpNumOfElements, mpiSource, pstatus.computingIndex);
				#endif
				#if DEBUG >= 4
					printf("Master: Saving tmpResults: ");
					for (int i = 0; i < tmpNumOfElements; i++) {
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
					printf("Master: Slave %d benchmark: %d elements, %f ms\n", mpiSource, pstatus.assignedElements, completionTime);
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

						#if DEBUG >= 3
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
					printf("Master: Slave %d exiting\n", mpiSource);
				#endif

				if(pstatus.assignedElements != 0){
					printf("[E] Slave %d exited with %d assigned elements!!\n", mpiSource, pstatus.assignedElements);
				}

				pstatus.computingIndex = totalElements;
				pstatus.finished = true;

				finished++;

				break;
		}

		MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses);
	}

	delete[] tmpResults;
	delete[] tmpToCalculate;
	free(processStatus);

	tmpNumOfElements = 0;
	MPI_Bcast(&tmpNumOfElements, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void ParallelFramework::coordinatorThread(ProcessingThreadInfo* pti, int numOfThreads){
	sem_t* semResults = pti[0].semResults;

	int numOfElements, elementsPerThread, tmp;
	unsigned long maxBatchSize = getDefaultCPUBatchSize();		// TODO: Also consider GPUs
	unsigned long *startPointIdx = new unsigned long[parameters->D];
	unsigned long allocatedElements = 0;
	RESULT_TYPE* results = nullptr;
	MPI_Status status;

	while(true){
		// Send READY signal to master
		#if DEBUG >= 2
			printf("Coordinator: Sending READY...\n");
		#endif
		MPI_Send(nullptr, 0, MPI_INT, 0, TAG_READY, MPI_COMM_WORLD);
		MPI_Send(&maxBatchSize, 1, MPI_UNSIGNED_LONG, 0, TAG_MAX_DATA_COUNT, MPI_COMM_WORLD);

		// Receive a batch of data from master
		MMPI_Recv(&numOfElements, 1, MPI_INT, 0, TAG_DATA_COUNT, MPI_COMM_WORLD, &status);
		MMPI_Recv(startPointIdx, parameters->D, MPI_UNSIGNED_LONG, 0, TAG_DATA, MPI_COMM_WORLD, &status);

		#if DEBUG >= 3
			printf("Coordinator: Received %d elements starting at: ", numOfElements);
			for(int i=0; i<parameters->D; i++)
				printf("%ld ", startPointIdx[i]);
			printf("\n");
		#elif DEBUG >= 2
			printf("Coordinator: Received %d elements\n", numOfElements);
		#endif
		// If no results, break
		if(numOfElements == 0)
			break;

		// Make sure we have enough allocated memory for the results
		if(numOfElements > allocatedElements){
			allocatedElements = numOfElements;

			if(results != nullptr) cudaFreeHost(results);
			cudaHostAlloc(&results, allocatedElements * sizeof(RESULT_TYPE), cudaHostAllocPortable);
		}

		// Split the data into numOfThreads pieces
		elementsPerThread = numOfElements / numOfThreads;

		#if DEBUG >= 2
			printf("Coordinator: Split data into %d elements for each thread\n", elementsPerThread);
			printf("Coordinator: Posting worker threads...\n");
		#endif

		int skip = 0;
		for(int i=0; i<numOfThreads; i++){
			tmp = 0;
			if(i==0){
				// Set the starting point as the sessions starting point
				memcpy(pti[i].startPointIdx, startPointIdx, parameters->D * sizeof(unsigned long));

				// Set numOfElements as elementsPerThread, or all of them if elementsPerThread==0
				pti[i].numOfElements = elementsPerThread==0 ? numOfElements : elementsPerThread;

				// Set results as the start of global results
				pti[i].results = results;
			}else{
				// Set the starting point as the starting point of the previous thread + numOfElements of the previous thread
				addToIdxVector(pti[i-1].startPointIdx, pti[i].startPointIdx, pti[i-1].numOfElements, &tmp);

				// Set the numOfelements as elementsPerThread, or all the remaining if we are at the last thread
				pti[i].numOfElements = i==numOfThreads-1 ? numOfElements-skip : elementsPerThread;

				// Set results as the results of the previous thread + numOfElements of the previous thread
				pti[i].results = pti[i-1].results + pti[i-1].numOfElements;
			}

			#if DEBUG >=2
				printf("Coordinator: Thread %d -> Assigning %d elements starting at: ", i, pti[i].numOfElements);
				for(int j=0; j < (parameters->D); j++){
					printf("%ld ", pti[i].startPointIdx[j]);
				}
				printf("\n");
			#endif

			if(tmp!=0){
				printf("[E] Coordinator: addToIdxVector for thread %d returned overflow = %d\n", i, tmp);
				break;
			}

			skip += pti[i].numOfElements;
		}

		// Start all the worker threads
		for(int i=0; i<numOfThreads; i++){
			sem_post(&pti[i].semData);
		}

		#if DEBUG >= 2
			printf("Coordinator: Waiting for results...\n");
		#endif
		// Wait for all worker threads to finish their work
		for(int i=0; i<numOfThreads; i++){
			sem_wait(semResults);
		}

		float linTime = 0;
		for(int i=0; i<numOfThreads; i++){
			if(parameters->benchmark){
				printf("Coordinator: Thread %d time: %f ms\n", pti[i].id, pti[i].stopwatch.getMsec());
			}

			linTime += pti[i].stopwatch.getMsec();
		}

		for(int i=0; i<numOfThreads; i++){
			pti[i].ratio = 1 - (pti[i].stopwatch.getMsec() / linTime);

			#if DEBUG >= 1
				printf("Coordinator: Adjusting thread %d ratio to %f\n", pti[i].id, pti[i].ratio);
			#endif
		}

		// Send all results to master
		#if DEBUG >= 2
			printf("Coordinator: Sending data to master...\n");
		#endif
		MPI_Send(nullptr, 0, MPI_INT, 0, TAG_RESULTS, MPI_COMM_WORLD);
		MPI_Send(results, numOfElements, RESULT_MPI_TYPE, 0, TAG_RESULTS_DATA, MPI_COMM_WORLD);
	}

	// Notify about exiting
	MPI_Send(nullptr, 0, MPI_INT, 0, TAG_EXITING, MPI_COMM_WORLD);

	// Notify worker threads to finish
	for(int i=0; i<numOfThreads; i++){
		pti[i].numOfElements = 0;
		sem_post(&pti[i].semData);
	}

	delete[] startPointIdx;
	cudaFreeHost(results);
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

	addToIdxVector(toSendVector, toSendVector, batchSize, nullptr);

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
		dst[i] = (int) round(abs(limits[i].lowerLimit - point[i]) / limits[i].step);		// TODO: 1.9999997 will round to 2, verify correctness
	}
}
long ParallelFramework::getIndexFromIndices(unsigned long* pointIdx) {
	unsigned int i;
	long index = 0;

	for (i = 0; i < parameters->D; i++) {
		// Increase index by i*(index-steps for this dimension)
		index += pointIdx[i] * idxSteps[i];
	}

	return index;
}

void ParallelFramework::addToIdxVector(unsigned long* start, unsigned long* result, int num, int* overflow){
    unsigned int i;
	unsigned int newIndex;
	unsigned int carry = num;

	for (i = 0; i < parameters->D; i++) {
		newIndex = (start[i] + carry) % limits[i].N;
		carry = (start[i] + carry) / limits[i].N;

		result[i] = newIndex;

		if(carry == 0 && start==result)
			break;
		// else we need to write the rest of the indices from start to result
	}

	if(overflow != nullptr){
		*overflow = carry;
	}
}
