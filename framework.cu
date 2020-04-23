#include "framework.h"

using namespace std;

ParallelFramework::ParallelFramework(Limit* limits, ParallelFrameworkParameters& parameters) {
	unsigned int i;
	valid = false;

	// Verify parameters
	if (parameters.D == 0 || parameters.D>MAX_DIMENSIONS) {
		cout << "[Init] [E] Dimension must be between 1 and " << MAX_DIMENSIONS << endl;
		return;
	}

	for (i = 0; i < parameters.D; i++) {
		if (limits[i].lowerLimit > limits[i].upperLimit) {
			cout << "[Init] [E] Limits for dimension " << i << ": Lower limit can't be higher than upper limit" << endl;
			return;
		}

		if (limits[i].N == 0) {
			cout << "[Init] [E] Limits for dimension " << i << ": N must be > 0" << endl;
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
		finalResults = new RESULT_TYPE[totalElements];		// Uninitialized
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
	delete [] finalResults;
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
	float totalScore;
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses);
	int numOfSlaves = numOfProcesses - 1;

	SlaveProcessInfo* slaveProcessInfo = new SlaveProcessInfo[numOfSlaves];
	#define pinfo (slaveProcessInfo[mpiSource-1])

	unsigned long allocatedElements = parameters->batchSize;				// Number of allocated elements for results
	RESULT_TYPE* tmpResults = new RESULT_TYPE[allocatedElements];
	unsigned long* tmpToCalculate = new unsigned long[parameters->D];
	int tmpNumOfElements;	// This needs to be int because of MPI

	for(int i=0; i<numOfSlaves; i++){
		// TODO: Add any more initializations
		slaveProcessInfo[i].id = i + 1;
		slaveProcessInfo[i].currentBatchSize = parameters->batchSize;
		slaveProcessInfo[i].computingIndex = 0;
		slaveProcessInfo[i].assignedElements = 0;
		slaveProcessInfo[i].jobsCompleted = 0;
		slaveProcessInfo[i].elementsCalculated = 0;
		slaveProcessInfo[i].finished = false;
		slaveProcessInfo[i].lastScore = -1;
		slaveProcessInfo[i].ratio = (float)1/numOfSlaves;
		slaveProcessInfo[i].stopwatch.reset();
	}

	while (totalReceived < totalElements || finished < numOfSlaves) {
		// Receive request from any worker thread
		#ifdef DBG_MPI_STEPS
			printf("[%d] Master: Waiting for signal...\n", rank);
		#endif
		fflush(stdout);

		MMPI_Recv(nullptr, 0, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		mpiSource = status.MPI_SOURCE;

		#ifdef DBG_MPI_STEPS
			printf("[%d] Master: Received %d from %d\n", rank, status.MPI_TAG, mpiSource);
		#endif

		switch(status.MPI_TAG){
			case TAG_READY:
				// Receive the maximum batch size reported by the slave process
				MMPI_Recv(&pinfo.maxBatchSize, 1, MPI_UNSIGNED_LONG, mpiSource, TAG_MAX_DATA_COUNT, MPI_COMM_WORLD, &status);

				// Get next data batch to calculate
				//getDataChunk(pinfo.currentBatchSize, tmpToCalculate, &tmpNumOfElements);
				getDataChunk(min((int)(pinfo.ratio * parameters->batchSize), (int)pinfo.maxBatchSize), tmpToCalculate, &tmpNumOfElements);
				pinfo.computingIndex = getIndexFromIndices(tmpToCalculate);

				#ifdef DBG_MPI_STEPS
					printf("[%d] Master: Sending %d elements to %d with index %d\n", rank, tmpNumOfElements, mpiSource, pinfo.computingIndex);
				#endif
				#ifdef DBG_DATA
					printf("[%d] Master: Sending data to %d: ", rank, mpiSource);
					for (unsigned int i = 0; i < parameters->D; i++) {
						printf("%d ", tmpToCalculate[i]);
					}
					printf("\n");
				#endif

				// Send the batch to the slave process
				MPI_Send(&tmpNumOfElements, 1, MPI_INT, mpiSource, TAG_DATA_COUNT, MPI_COMM_WORLD);
				MPI_Send(tmpToCalculate, parameters->D, MPI_UNSIGNED_LONG, mpiSource, TAG_DATA, MPI_COMM_WORLD);

				// Update details for process
				pinfo.stopwatch.start();
				pinfo.assignedElements = tmpNumOfElements;
				break;

			case TAG_RESULTS:
				// Receive the results
				MMPI_Recv(tmpResults, pinfo.maxBatchSize, RESULT_MPI_TYPE, mpiSource, TAG_RESULTS_DATA, MPI_COMM_WORLD, &status);

				// Find the length of the results
				MPI_Get_count(&status, RESULT_MPI_TYPE, &tmpNumOfElements);	// This is equal to pinfo.assignedElements

				#ifdef DBG_MPI_STEPS
					printf("[%d] Master: Saving %ld results from slave %d to results[%ld]...\n", rank, tmpNumOfElements, mpiSource, pinfo.computingIndex);
				#endif
				#ifdef DBG_RESULTS
					printf("[%d] Master: Saving tmpResults: ", rank);
					for (int i = 0; i < tmpNumOfElements; i++) {
						printf("%f ", tmpResults[i]);
					}
					printf(" at %d\n", pinfo.computingIndex);
				#endif

				// Update pinfo
				pinfo.jobsCompleted++;
				pinfo.elementsCalculated += tmpNumOfElements;
				pinfo.stopwatch.stop();
				pinfo.lastScore = pinfo.assignedElements / pinfo.stopwatch.getMsec();
				pinfo.lastAssignedElements = pinfo.assignedElements;

				this->totalReceived += tmpNumOfElements;

				// Print benchmark results
				if (parameters->benchmark) {
					printf("[%d] Master: Slave %d benchmark: %d elements, %f ms\n", rank, mpiSource, pinfo.assignedElements, pinfo.stopwatch.getMsec());
				}

				if(parameters->slaveBalancing && numOfSlaves > 1){
					// Check other scores and calculate the sum
					totalScore = 0;
					for(int i=0; i<numOfSlaves; i++){
						totalScore += slaveProcessInfo[i].lastScore;

						if(slaveProcessInfo[i].lastScore < 0){
							// Either score has not been set, or got negative time
							totalScore = -1;
							break;
						}
					}

					// If all the processes have a real score (has been set and not negative)
					if(totalScore > 0){
						for(int i=0; i<numOfSlaves; i++){
							slaveProcessInfo[i].ratio = slaveProcessInfo[i].lastScore / totalScore;
							#ifdef DBG_RATIO
								printf("[%d] Master: Adjusting slave %d ratio = %f\n", rank, slaveProcessInfo[i].id, slaveProcessInfo[i].ratio);
							#endif
						}
					}else{
						#ifdef DBG_RATIO
							printf("[%d] Master: Skipping ratio adjustment\n", rank);
						#endif
					}
				}

				// Reset pinfo
				pinfo.assignedElements = 0;
				pinfo.stopwatch.reset();

				// Copy the received results to finalResults
				if( ! (parameters->benchmark))
					memcpy(&finalResults[pinfo.computingIndex], tmpResults, tmpNumOfElements*sizeof(RESULT_TYPE));

				break;

			case TAG_EXITING:
				#ifdef DBG_MPI_STEPS
					printf("[%d] Master: Slave %d exiting\n", rank, mpiSource);
				#endif

				if(pinfo.assignedElements != 0){
					printf("[%d] [E] Master: Slave %d exited with %d assigned elements!!\n", rank, mpiSource, pinfo.assignedElements);
				}

				pinfo.computingIndex = totalElements;
				pinfo.finished = true;

				finished++;

				break;
		}
	}

	// Notify all slave processes to finish
	tmpNumOfElements = 0;
	MPI_Bcast(&tmpNumOfElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

	delete[] tmpResults;
	delete[] tmpToCalculate;
	delete[] slaveProcessInfo;
}

void ParallelFramework::coordinatorThread(ComputeThreadInfo* cti, int numOfThreads){
	sem_t* semResults = cti[0].semResults;

	int numOfElements, tmp;
	unsigned long maxBatchSize = min(getDefaultCPUBatchSize(), getDefaultGPUBatchSize());
	unsigned long *startPointIdx = new unsigned long[parameters->D];
	unsigned long allocatedElements = 0;
	RESULT_TYPE* localResults = nullptr;
	MPI_Status status;

	#ifdef DBG_START_STOP
		printf("[%d] Coordinator: Max batch size: %d\n", rank, maxBatchSize);
	#endif

	while(true){
		// Send READY signal to master
		#ifdef DBG_MPI_STEPS
			printf("[%d] Coordinator: Sending READY...\n", rank);
		#endif
		MPI_Send(nullptr, 0, MPI_INT, 0, TAG_READY, MPI_COMM_WORLD);
		MPI_Send(&maxBatchSize, 1, MPI_UNSIGNED_LONG, 0, TAG_MAX_DATA_COUNT, MPI_COMM_WORLD);

		// Receive a batch of data from master
		MMPI_Recv(&numOfElements, 1, MPI_INT, 0, TAG_DATA_COUNT, MPI_COMM_WORLD, &status);
		MMPI_Recv(startPointIdx, parameters->D, MPI_UNSIGNED_LONG, 0, TAG_DATA, MPI_COMM_WORLD, &status);

		#ifdef DBG_DATA
			printf("[%d] Coordinator: Received %d elements starting at: ", rank, numOfElements);
			for(int i=0; i<parameters->D; i++)
				printf("%ld ", startPointIdx[i]);
			printf("\n");
		#endif
		#ifdef DBG_MPI_STEPS
			printf("[%d] Coordinator: Received %d elements\n", rank, numOfElements);
		#endif

		// If no results, break
		if(numOfElements == 0)
			break;

		// Make sure we have enough allocated memory for localResults
		if(numOfElements > allocatedElements){
			#ifdef DBG_MEMORY
				printf("[%d] Coordinator: Allocating more memory for localResults: %d (0x%x) -> ", rank, allocatedElements, localResults);
			#endif

			allocatedElements = numOfElements;
			// if(localResults != nullptr) cudaFreeHost(localResults);		// TODO: This causes a problem when running without GPU
			// cudaHostAlloc(&localResults, allocatedElements * sizeof(RESULT_TYPE), cudaHostAllocPortable);
			localResults = (RESULT_TYPE*) realloc(localResults, allocatedElements * sizeof(RESULT_TYPE));

			#ifdef DBG_MEMORY
				printf("%d (0x%x)\n", allocatedElements, localResults);
			#endif
		}

		// Split the data into pieces for each thread
		int total = 0;
		for(int i=0; i<numOfThreads; i++){
			cti[i].numOfElements = numOfElements * cti[i].ratio;
			total += cti[i].numOfElements;
		}
		if(total < numOfElements){
			// Something was left out, assign it to first thread
			cti[0].numOfElements += numOfElements - total;
		}else if(total > numOfElements){
			// More elements were given, remove from the first threads
			for(int i=0; i<numOfThreads; i++){
				// If thread i has enough elements, remove all the extra from it
				if(cti[i].numOfElements > total-numOfElements){
					cti[i].numOfElements -= total-numOfElements;
					total = numOfElements;
					break;
				}

				// else assign 0 elements to it and move it to the next thread
				total -= cti[i].numOfElements;
				cti[i].numOfElements = 0;
			}
		}

		// Assign work to the worker threads
		for(int i=0; i<numOfThreads; i++){
			tmp = 0;

			if(i==0){
				// Set the starting point as the sessions starting point
				memcpy(cti[i].startPointIdx, startPointIdx, parameters->D * sizeof(unsigned long));

				// Set results as the start of global results
				cti[i].results = localResults;
			}else{
				// Set the starting point as the starting point of the previous thread + numOfElements of the previous thread
				addToIdxVector(cti[i-1].startPointIdx, cti[i].startPointIdx, cti[i-1].numOfElements, &tmp);

				// Set results as the results of the previous thread + numOfElements of the previous thread (compiler takes into account the size of RESULT_TYPE when adding an int)
				cti[i].results = cti[i-1].results + cti[i-1].numOfElements;
			}

			#ifdef DBG_QUEUE
				printf("[%d] Coordinator: Thread %d -> Assigning %d elements starting at: ", rank, i, cti[i].numOfElements);
				for(int j=0; j < (parameters->D); j++){
					printf("%ld ", cti[i].startPointIdx[j]);
				}
				printf(" with results at 0x%x\n", cti[i].results);
			#endif

			if(tmp!=0){
				printf("[%d] [E] Coordinator: addToIdxVector for thread %d returned overflow = %d\n", rank, i, tmp);
				break;
			}
		}

		// Start all the worker threads
		for(int i=0; i<numOfThreads; i++){
			sem_post(&cti[i].semData);
		}

		#ifdef DBG_MPI_STEPS
			printf("[%d] Coordinator: Waiting for results...\n", rank);
		#endif
		// Wait for all worker threads to finish their work
		for(int i=0; i<numOfThreads; i++){
			sem_wait(semResults);
		}

		// Start 2 threads: 1 to adjust the ratios, one to send the results to master
		omp_set_nested(1);
		#pragma omp parallel num_threads(parameters->threadBalancing && numOfThreads>1 ? 2 : 1)
		{
			if(omp_get_thread_num() == 1){	// Will run if parameters->threadBalancing == true
				// Calculate a score for each thread (=numOfElements/time)
				float tmpScore;
				float totalScore = 0;
				for(int i=0; i<numOfThreads; i++){
					if(parameters->benchmark){
						printf("[%d] Coordinator: Thread %d time: %f ms\n", rank, cti[i].id, cti[i].stopwatch.getMsec());
					}

					tmpScore = cti[i].numOfElements / cti[i].stopwatch.getMsec();
					totalScore += tmpScore;

					if(tmpScore < 0){
						totalScore = -1;
						break;
					}
				}

				// Adjust the ratio for each thread
				if(totalScore > 0){
					for(int i=0; i<numOfThreads; i++){
						cti[i].ratio = cti[i].numOfElements / (totalScore * cti[i].stopwatch.getMsec());

						#ifdef DBG_RATIO
							printf("[%d] Coordinator: Adjusting thread %d ratio to %f\n", rank, cti[i].id, cti[i].ratio);
						#endif
					}
				}else{
					printf("[%d] [E] Coordinator: Got negative time, skipping ratio correction\n", rank);
				}

			}else{
				// Send all results to master
				#ifdef DBG_MPI_STEPS
					printf("[%d] Coordinator: Sending data to master...\n", rank);
				#endif
				MPI_Send(nullptr, 0, MPI_INT, 0, TAG_RESULTS, MPI_COMM_WORLD);
				MPI_Send(localResults, numOfElements, RESULT_MPI_TYPE, 0, TAG_RESULTS_DATA, MPI_COMM_WORLD);
			}
		}

		// Reset the stopwatches
		for(int i=0; i<numOfThreads; i++)
			cti[i].stopwatch.reset();
	}

	// Notify about exiting
	MPI_Send(nullptr, 0, MPI_INT, 0, TAG_EXITING, MPI_COMM_WORLD);

	// Notify worker threads to finish
	for(int i=0; i<numOfThreads; i++){
		cti[i].numOfElements = -1;
		sem_post(&cti[i].semData);
	}

	delete[] startPointIdx;
	//cudaFreeHost(localResults);
	free(localResults);
}

void ParallelFramework::getDataChunk(unsigned long batchSize, unsigned long* toCalculate, int* numOfElements) {
	if (totalSent >= totalElements) {
		*numOfElements = 0;
		return;
	}

	// Adjust batchSize if it's more than the available elements
	if (totalElements - totalSent < batchSize)
		batchSize = totalElements - totalSent;

	// Copy toSendVector to the output
	memcpy(toCalculate, toSendVector, parameters->D * sizeof(long));

	// Set output numOfElements
	*numOfElements = batchSize;

	// Increase toSendVector for next call here
	addToIdxVector(toSendVector, toSendVector, batchSize, nullptr);
	totalSent += batchSize;
}

RESULT_TYPE* ParallelFramework::getResults() {
	return finalResults;
}
void ParallelFramework::getIndicesFromPoint(DATA_TYPE* point, unsigned long* dst) {
	unsigned int i;

	for (i = 0; i < parameters->D; i++) {
		if (point[i] < limits[i].lowerLimit || point[i] >= limits[i].upperLimit) {
			printf("Result query for out-of-bounds point\n");
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
