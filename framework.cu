#include <limits.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>

#include "framework.h"

using namespace std;

ParallelFramework::ParallelFramework(Limit* limits, ParallelFrameworkParameters& parameters) {
	unsigned int i;
	valid = false;

	// Verify parameters
	if (parameters.D == 0 || parameters.D>MAX_DIMENSIONS) {
		cout << "[Init] Error: Dimension must be between 1 and " << MAX_DIMENSIONS << endl;
		return;
	}

	for (i = 0; i < parameters.D; i++) {
		if (limits[i].lowerLimit > limits[i].upperLimit) {
			cout << "[Init] Error: Limits for dimension " << i << ": Lower limit can't be higher than upper limit" << endl;
			return;
		}

		if (limits[i].N == 0) {
			cout << "[Init] Error: Limits for dimension " << i << ": N must be > 0" << endl;
			return;
		}
	}

	idxSteps = new unsigned long long[parameters.D];
	idxSteps[0] = 1;
	for (i = 1; i < parameters.D; i++) {
		idxSteps[i] = idxSteps[i - 1] * limits[i-1].N;
	}

	for (i = 0; i < parameters.D; i++) {
		limits[i].step = abs(limits[i].upperLimit - limits[i].lowerLimit) / limits[i].N;
	}

	totalReceived = 0;
	totalSent = 0;
	totalElements = (unsigned long long)(idxSteps[parameters.D - 1]) * (unsigned long long)(limits[parameters.D - 1].N);
	if(! (parameters.benchmark)){
		if(parameters.resultSaveType == SAVE_TYPE_ALL){
			if(parameters.saveFile == nullptr){
				// No saveFile given, save everything in memory
				finalResults = new RESULT_TYPE[totalElements];		// Uninitialized
			}else{
				// Open save file
				saveFile = open(parameters.saveFile, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
				if(saveFile == -1){
					fatal("open failed");
				}

				// Enlarge the file
				if(ftruncate(saveFile, totalElements * sizeof(RESULT_TYPE)) == -1){
					fatal("ftruncate failed");
				}

				// Map the save file in memory
				finalResults = (RESULT_TYPE*) mmap(nullptr, totalElements * sizeof(RESULT_TYPE), PROT_WRITE, MAP_SHARED, saveFile, 0);
				if(finalResults == MAP_FAILED){
					fatal("mmap failed");
				}

				#ifdef DBG_MEMORY
					printf("[Init] finalResults: 0x%lx\n", finalResults);
				#endif
			}
		}// else listResults will be allocated through realloc when they are needed
	}

	toSendVector = new unsigned long[parameters.D];
	for (i = 0; i < parameters.D; i++) {
		toSendVector[i] = 0;
	}

	this->limits = limits;
	this->parameters = &parameters;

	if (this->parameters->batchSize == 0)
		this->parameters->batchSize = totalElements;

	valid = true;
}

ParallelFramework::~ParallelFramework() {
	delete [] idxSteps;
	if(parameters->saveFile == nullptr){
		delete [] finalResults;
	}else{
		// Unmap the save file
		munmap(finalResults, totalElements * sizeof(RESULT_TYPE));

		// Close the file
		close(saveFile);
	}
	delete [] toSendVector;
	if(listResults != NULL){
		free(listResults);
		listResultsSaved = 0;
	}
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
	int t, eta;
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses);
	int numOfSlaves = numOfProcesses - 1;
	Stopwatch masterStopwatch;

	SlaveProcessInfo* slaveProcessInfo = new SlaveProcessInfo[numOfSlaves];
	#define pinfo (slaveProcessInfo[mpiSource-1])

	RESULT_TYPE* tmpResults = new RESULT_TYPE[parameters->batchSize];
	DATA_TYPE* tmpResultsList = new DATA_TYPE[parameters->batchSize * parameters->D];
	unsigned long* tmpToCalculate = new unsigned long[parameters->D];
	int tmpNumOfElements, tmpNumOfPoints;	// These need to be int because of MPI

	#ifdef DBG_TIME
		Stopwatch sw;
	#endif

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
	}

	masterStopwatch.start();
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
				#ifdef DBG_TIME
					sw.start();
				#endif
				// Receive the maximum batch size reported by the slave process
				MMPI_Recv(&pinfo.maxBatchSize, 1, MPI_UNSIGNED_LONG, mpiSource, TAG_MAX_DATA_COUNT, MPI_COMM_WORLD, &status);

				// For the first batches, use low batch size so the process can optimize its computeThread scores early
				if(pinfo.jobsCompleted < SLOW_START_LIMIT){
					pinfo.maxBatchSize = min(pinfo.maxBatchSize, (unsigned long) (SLOW_START_BATCH_SIZE_BASE * pow(2, pinfo.jobsCompleted)));
					#ifdef DBG_RATIO
						printf("[%d] Master: Setting temporary maxBatchSize=%ld for slave %d\n", rank, pinfo.maxBatchSize, mpiSource);
					#endif
				}

				// Get next data batch to calculate
				getDataChunk(parameters->resultSaveType == SAVE_TYPE_ALL ? min((int)(pinfo.ratio * parameters->batchSize), (int)pinfo.maxBatchSize) : (int)pinfo.maxBatchSize, tmpToCalculate, &tmpNumOfElements);
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

				#ifdef DBG_TIME
					sw.stop();
					printf("[%d] Master: Benchmark: Time for TAG_READY: %f ms\n", rank, sw.getMsec());
				#endif

				break;

			case TAG_RESULTS:
				#ifdef DBG_TIME
					sw.start();
				#endif

				// Receive the results
				if(parameters->resultSaveType == SAVE_TYPE_ALL){
					MMPI_Recv(tmpResults, pinfo.maxBatchSize, RESULT_MPI_TYPE, mpiSource, TAG_RESULTS_DATA, MPI_COMM_WORLD, &status);

					// Find the length of the results
					MPI_Get_count(&status, RESULT_MPI_TYPE, &tmpNumOfElements);	// This is equal to pinfo.assignedElements
				}else{
					MMPI_Recv(tmpResultsList, pinfo.maxBatchSize * parameters->D, DATA_MPI_TYPE, mpiSource, TAG_RESULTS_DATA, MPI_COMM_WORLD, &status);

					// Find the number of points in list
					MPI_Get_count(&status, DATA_MPI_TYPE, &tmpNumOfPoints);

					// MPI_Get_count returned the count of DATA_TYPE elements received, so divide with D to get the count of points
					tmpNumOfPoints /= parameters->D;
				}

				this->totalReceived += tmpNumOfElements;

				masterStopwatch.stop();
				t = masterStopwatch.getMsec()/1000;
				eta = t * ((float)totalElements/totalReceived) - t;

				printf("Progress: %ld/%ld, %.2f %%", this->totalReceived, this->totalElements, ((float)this->totalReceived / this->totalElements)*100);

				if(t < 3600)	printf(", Elapsed time: %02d:%02d", t/60, t%60);
				else			printf(", Elapsed time: %02d:%02d:%02d", t/3600, (t%3600)/60, t%60);

				if(eta < 3600)	printf(", ETA: %02d:%02d\n", eta/60, eta%60);
				else			printf(", ETA: %02d:%02d:%02d\n", eta/3600, (eta%3600)/60, eta%60);

				#ifdef DBG_MPI_STEPS
					if(parameters->resultSaveType == SAVE_TYPE_ALL)
						printf("[%d] Master: Saving %ld results from slave %d to finalResults[%ld]...\n", rank, tmpNumOfElements, mpiSource, pinfo.computingIndex);
					else
						printf("[%d] Master: Saving %ld points from slave %d to listResults[%ld]...\n", rank, tmpNumOfPoints, mpiSource, listResultsSaved);

				#endif
				#ifdef DBG_RESULTS
					if(parameters->resultSaveType == SAVE_TYPE_ALL){
						printf("[%d] Master: Saving tmpResults: ", rank);
						for (int i = 0; i < tmpNumOfElements; i++) {
							printf("%f ", tmpResults[i]);
						}
						printf(" at %d\n", pinfo.computingIndex);
					}else{
						printf("[%d] Master: Saving tmpResultsList: ", rank);
						for (int i = 0; i < tmpNumOfPoints; i++){
							printf("[ ");
							for(int j=0; j<parameters->D; j++){
								printf("%f ", tmpResultsList[i*parameters->D + j]);
							}
							printf("]");
						}
						printf(" at %d\n", listResultsSaved);
					}
				#endif

				// Copy the received results to finalResults or listResults
				if( ! (parameters->benchmark)){
					if(parameters->resultSaveType == SAVE_TYPE_ALL){
						memcpy(&finalResults[pinfo.computingIndex], tmpResults, tmpNumOfElements*sizeof(RESULT_TYPE));
					}else if(tmpNumOfPoints > 0){
						// Reallocate listResults
						listResultsSaved += tmpNumOfPoints;
						listResults = (DATA_TYPE*) realloc(listResults, listResultsSaved * parameters->D * sizeof(DATA_TYPE));
						if(listResults == nullptr){
							fatal("Can't allocate memory for listResults");
						}

						// Append the received data in listResults
						memcpy(&listResults[(listResultsSaved - tmpNumOfPoints) * parameters->D], tmpResultsList, tmpNumOfPoints * parameters->D * sizeof(DATA_TYPE));
					}
				}

				// Update pinfo
				pinfo.jobsCompleted++;
				pinfo.elementsCalculated += tmpNumOfElements;
				pinfo.stopwatch.stop();
				pinfo.lastScore = pinfo.assignedElements / pinfo.stopwatch.getMsec();
				pinfo.lastAssignedElements = pinfo.assignedElements;

				// Print benchmark results
				if (parameters->benchmark) {
					printf("[%d] Master: Slave %d benchmark: %d elements, %f ms\n", rank, mpiSource, pinfo.assignedElements, pinfo.stopwatch.getMsec());
				}

				if(parameters->slaveBalancing && numOfSlaves > 1){
					// Check other scores and calculate the sum
					totalScore = 0;
					for(int i=0; i<numOfSlaves; i++){
						totalScore += slaveProcessInfo[i].lastScore;

						if(slaveProcessInfo[i].lastScore <= 0){
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

				#ifdef DBG_TIME
					sw.stop();
					printf("[%d] Master: Benchmark: Time for TAG_RESULTS: %f ms\n", rank, sw.getMsec());
				#endif

				break;

			case TAG_EXITING:
				#ifdef DBG_MPI_STEPS
					printf("[%d] Master: Slave %d exiting\n", rank, mpiSource);
				#endif

				if(pinfo.assignedElements != 0){
					printf("[%d] Master: Error: Slave %d exited with %d assigned elements!!\n", rank, mpiSource, pinfo.assignedElements);
				}

				pinfo.computingIndex = totalElements;
				pinfo.finished = true;

				finished++;

				break;
		}
	}

	delete[] tmpResults;
	delete[] tmpToCalculate;
	delete[] slaveProcessInfo;
}

void ParallelFramework::coordinatorThread(ComputeThreadInfo* cti, int numOfThreads, Model* model){
	sem_t* semResults = cti[0].semResults;
	int* globalListIndexPtr = cti[0].listIndexPtr;

	int numOfElements, carry;
	unsigned long maxBatchSize;
	unsigned long *startPointIdx = new unsigned long[parameters->D];
	unsigned long allocatedElements = 0;
	RESULT_TYPE* localResults = nullptr;
	MPI_Status status;

	#ifdef DBG_TIME
		Stopwatch sw;
		float time_data, time_split, time_assign, time_start, time_wait, time_scores, time_results;
	#endif

	if(parameters->processingType == PROCESSING_TYPE_CPU)
		maxBatchSize = getDefaultCPUBatchSize();
	else if(parameters->processingType == PROCESSING_TYPE_GPU)
		maxBatchSize = getDefaultGPUBatchSize();
	else
		maxBatchSize = min(getDefaultCPUBatchSize(), getDefaultGPUBatchSize());

	maxBatchSize = min((unsigned long)parameters->batchSize, (unsigned long)maxBatchSize);

	if(maxBatchSize*parameters->D > INT_MAX && parameters->resultSaveType == SAVE_TYPE_LIST){
		maxBatchSize = (INT_MAX - parameters->D) / parameters->D;
	}else if(maxBatchSize > INT_MAX && parameters->resultSaveType == SAVE_TYPE_ALL){
		maxBatchSize = INT_MAX;
	}


	#ifdef DBG_START_STOP
		printf("[%d] Coordinator: Max batch size: %d\n", rank, maxBatchSize);
	#endif

	while(true){
		// Send READY signal to master
		#ifdef DBG_MPI_STEPS
			printf("[%d] Coordinator: Sending READY...\n", rank);
		#endif
		#ifdef DBG_TIME
			sw.start();
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
			localResults = (RESULT_TYPE*) realloc(localResults, allocatedElements * sizeof(RESULT_TYPE));
			if(localResults == nullptr){
				fatal("Can't allocate memory for localResults");
			}

			#ifdef DBG_MEMORY
				printf("%d (0x%x)\n", allocatedElements, localResults);
			#endif
		}

		#ifdef DBG_TIME
			sw.stop();
			time_data = sw.getMsec();
			sw.start();
		#endif

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

				// else assign 0 elements to it and move to the next thread
				total -= cti[i].numOfElements;
				cti[i].numOfElements = 0;
			}
		}

		#ifdef DBG_TIME
			sw.stop();
			time_split = sw.getMsec();
			sw.start();
		#endif

		// Assign work to the worker threads
		for(int i=0; i<numOfThreads; i++){
			carry = 0;

			if(i==0){
				// Set the starting point as the sessions starting point
				memcpy(cti[i].startPointIdx, startPointIdx, parameters->D * sizeof(unsigned long));

				// Set results as the start of global results
				cti[i].results = localResults;
			}else{
				// Set the starting point as the starting point of the previous thread + numOfElements of the previous thread
				addToIdxVector(cti[i-1].startPointIdx, cti[i].startPointIdx, cti[i-1].numOfElements, &carry);

				if(parameters->resultSaveType == SAVE_TYPE_ALL){
					// Set results as the results of the previous thread + numOfElements of the previous thread (compiler takes into account the size of RESULT_TYPE when adding an int)
					cti[i].results = cti[i-1].results + cti[i-1].numOfElements;
				}else{
					// All compute threads must have access to the same memory
					cti[i].results = localResults;
				}
			}

			#ifdef DBG_QUEUE
				printf("[%d] Coordinator: Thread %d -> Assigning %d elements starting at: ", rank, i, cti[i].numOfElements);
				for(int j=0; j < (parameters->D); j++){
					printf("%ld ", cti[i].startPointIdx[j]);
				}
				printf(" with results at 0x%x\n", cti[i].results);
			#endif

			if(carry!=0){
				printf("[%d] Coordinator: Error: addToIdxVector() for thread %d returned overflow = %d\n", rank, i, carry);
				break;
			}
		}

		#ifdef DBG_TIME
			sw.stop();
			time_assign = sw.getMsec();
			sw.start();
		#endif

		// Reset the global listIndex counter
		*globalListIndexPtr = 0;

		// Start all the worker threads
		for(int i=0; i<numOfThreads; i++){
			sem_post(&cti[i].semData);
		}

		#ifdef DBG_TIME
			sw.stop();
			time_start = sw.getMsec();
			sw.start();
		#endif

		#ifdef DBG_MPI_STEPS
			printf("[%d] Coordinator: Waiting for results...\n", rank);
		#endif
		// Wait for all worker threads to finish their work
		for(int i=0; i<numOfThreads; i++){
			sem_wait(semResults);
		}

		#ifdef DBG_TIME
			sw.stop();
			time_wait = sw.getMsec();
			sw.start();
		#endif

		#ifdef DBG_RESULTS
			if(parameters->resultSaveType == SAVE_TYPE_ALL){
				printf("[%d] Coordinator: Results: [", rank);
				for(int i=0; i<numOfElements; i++){
					printf("%f ", localResults[i]);
				}
				printf("]\n");
			}else{
				printf("[%d] Coordinator: Results (*globalListIndexPtr = %d):", rank, *globalListIndexPtr);
				for(int i=0; i<*globalListIndexPtr; i+=parameters->D){
					printf("[ ");
					for(int j=0; j<parameters->D; j++){
						printf("%f ", ((DATA_TYPE *)localResults)[i + j]);
					}
					printf("]");
				}
			}
		#endif

		// Calculate a score for each thread (=numOfElements/time)
		if(parameters->threadBalancing){
			float tmpScore;
			float totalScore = 0;
			for(int i=0; i<numOfThreads; i++){
				if(parameters->benchmark){
					printf("[%d] Coordinator: Thread %d time: %f ms\n", rank, cti[i].id, cti[i].stopwatch.getMsec());
				}

				tmpScore = cti[i].numOfElements / cti[i].stopwatch.getMsec();
				totalScore += tmpScore;

				if(tmpScore < 0){
					// Negative time
					totalScore = -1;
					break;
				}else if(cti[i].stopwatch.getMsec() < MIN_MS_FOR_RATIO_ADJUSTMENT){
					// Too fast execution
					totalScore = -2;
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
			}else if(totalScore == -1){
				printf("[%d] Coordinator: Error: Skipping ratio correction due to negative time\n", rank);
			}else{
				#ifdef DBG_RATIO
					printf("[%d] Coordinator: Skipping ratio correction due to fast execution\n", rank);
				#endif
			}

			#ifdef DBG_TIME
				sw.stop();
				time_scores = sw.getMsec();
				sw.start();
			#endif
		}

		// Send all results to master
		#ifdef DBG_MPI_STEPS
			printf("[%d] Coordinator: Sending data to master...\n", rank);
		#endif

		MPI_Send(nullptr, 0, MPI_INT, 0, TAG_RESULTS, MPI_COMM_WORLD);
		if(parameters->resultSaveType == SAVE_TYPE_ALL){
			// Send all the results
			MPI_Send(localResults, numOfElements, RESULT_MPI_TYPE, 0, TAG_RESULTS_DATA, MPI_COMM_WORLD);
		}else{
			// Send the list of points
			MPI_Send(localResults, *globalListIndexPtr, DATA_MPI_TYPE, 0, TAG_RESULTS_DATA, MPI_COMM_WORLD);
		}

		#ifdef DBG_TIME
			sw.stop();
			time_results = sw.getMsec();

			printf("[%d] Coordinator: Benchmark:\n", rank);
			printf("Time for data: %f ms\n", time_data);
			printf("Time for split: %f ms\n", time_split);
			printf("Time for assign: %f ms\n", time_assign);
			printf("Time for start: %f ms\n", time_start);
			printf("Time for wait: %f ms\n", time_wait);
			printf("Time for scores: %f ms\n", time_scores);
			printf("Time for results: %f ms\n", time_results);
		#endif
	}

	// Notify about exiting
	MPI_Send(nullptr, 0, MPI_INT, 0, TAG_EXITING, MPI_COMM_WORLD);

	// Notify worker threads to finish
	for(int i=0; i<numOfThreads; i++){
		cti[i].numOfElements = -1;
		sem_post(&cti[i].semData);
	}

	delete[] startPointIdx;
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
	if(parameters->resultSaveType != SAVE_TYPE_ALL){
		printf("Error: Can't get all results when resultSaveType is not SAVE_TYPE_ALL\n");
		return nullptr;
	}

	return finalResults;
}
DATA_TYPE* ParallelFramework::getList(int* length){
	if(parameters->resultSaveType != SAVE_TYPE_LIST){
		printf("Error: Can't get list results when resultSaveType is not SAVE_TYPE_LIST\n");
		if(length != nullptr)
			*length = -1;

		return nullptr;
	}

	if(length != nullptr)
		*length = listResultsSaved;

	return listResults;
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
unsigned long ParallelFramework::getIndexFromIndices(unsigned long* pointIdx) {
	unsigned int i;
	unsigned long index = 0;

	for (i = 0; i < parameters->D; i++) {
		// Increase index by i*(index-steps for this dimension)
		index += pointIdx[i] * idxSteps[i];
	}

	return index;
}
unsigned long ParallelFramework::getIndexFromPoint(DATA_TYPE* point){
	unsigned long* indices = new unsigned long[parameters->D];
	unsigned long index;

	getIndicesFromPoint(point, indices);
	index = getIndexFromIndices(indices);

	delete[] indices;
	return index;
}
void ParallelFramework::getPointFromIndex(unsigned long index, DATA_TYPE* result){
	for(int i=parameters->D - 1; i>=0; i--){
		int currentIndex = index / idxSteps[i];
		result[i] = limits[i].lowerLimit + currentIndex*limits[i].step;

		index = index % idxSteps[i];
	}
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
