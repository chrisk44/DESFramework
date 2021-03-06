#include <limits.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>

#include "framework.h"

using namespace std;

ParallelFramework::ParallelFramework(bool initMPI) {
	// The user might want to do the MPI Initialization. Useful when the framework is used more than once in a program.
	if(initMPI){
		// Initialize MPI
		MPI_Init(nullptr, nullptr);
	}
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	valid = false;
}
void ParallelFramework::init(Limit* _limits, ParallelFrameworkParameters& _parameters){
	unsigned int i;

	parameters = &_parameters;
	limits = _limits;

	// Verify parameters
	if (parameters->D == 0 || parameters->D>MAX_DIMENSIONS) {
		cout << "[Init] Error: Dimension must be between 1 and " << MAX_DIMENSIONS << endl;
		return;
	}

	for (i = 0; i < parameters->D; i++) {
		if (limits[i].lowerLimit > limits[i].upperLimit) {
			cout << "[Init] Error: Limits for dimension " << i << ": Lower limit can't be higher than upper limit" << endl;
			return;
		}

		if (limits[i].N == 0) {
			cout << "[Init] Error: Limits for dimension " << i << ": N must be > 0" << endl;
			return;
		}
	}

	if(parameters->dataPtr == nullptr && parameters->dataSize > 0){
		cout << "[Init] Error: dataPtr is null but dataSize is > 0" << endl;
		return;
	}

	if(parameters->overrideMemoryRestrictions && parameters->resultSaveType != SAVE_TYPE_LIST){
		cout << "[Init] Error: Can't override memory restrictions when saving as SAVE_TYPE_ALL" << endl;
		return;
	}

	idxSteps = new unsigned long long[parameters->D];
	idxSteps[0] = 1;
	for (i = 1; i < parameters->D; i++) {
		idxSteps[i] = idxSteps[i - 1] * limits[i-1].N;
	}

	for (i = 0; i < parameters->D; i++) {
		limits[i].step = abs(limits[i].upperLimit - limits[i].lowerLimit) / limits[i].N;
	}

	#ifdef DBG_DATA
		for(i=0; i < parameters->D; i++){
			printf("Dimension %d: Low=%lf, High=%lf, Step=%lf, N=%u, idxSteps=%llu\n", i, limits[i].lowerLimit, limits[i].upperLimit, limits[i].step, limits[i].N, idxSteps[i]);
		}
	#endif

	listResultsSaved = 0;
	totalReceived = 0;
	totalSent = 0;
	totalElements = (unsigned long long)(idxSteps[parameters->D - 1]) * (unsigned long long)(limits[parameters->D - 1].N);

	if(rank == 0){
		if(! (parameters->benchmark)){
			if(parameters->resultSaveType == SAVE_TYPE_ALL){
				if(parameters->saveFile == nullptr){
					// No saveFile given, save everything in memory
					finalResults = new RESULT_TYPE[totalElements];		// Uninitialized
				}else{
					// Open save file
					saveFile = open(parameters->saveFile, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
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

		if (this->parameters->batchSize == 0)
			this->parameters->batchSize = totalElements;
	}

	valid = true;
}

ParallelFramework::~ParallelFramework() {
	delete [] idxSteps;
	if(rank == 0){
		if(parameters->saveFile == nullptr){
			delete [] finalResults;
		}else{
			// Unmap the save file
			munmap(finalResults, totalElements * sizeof(RESULT_TYPE));

			// Close the file
			close(saveFile);
		}
		if(listResults != NULL){
			free(listResults);
			listResultsSaved = 0;
		}
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

	void* tmpResultsMem;
	if(parameters->overrideMemoryRestrictions){
		tmpResultsMem = malloc(getDefaultCPUBatchSize() * sizeof(DATA_TYPE));
	}else{
		tmpResultsMem = malloc(parameters->resultSaveType == SAVE_TYPE_ALL ? parameters->batchSize * sizeof(RESULT_TYPE) : parameters->batchSize * parameters->D * sizeof(DATA_TYPE));
	}
	if(tmpResultsMem == nullptr){
		printf("[%d] Master: Error: Can't allocate memory for tmpResultsMem\n", rank);
		exit(-1);
	}
	RESULT_TYPE* tmpResults = (RESULT_TYPE*) tmpResultsMem;
	DATA_TYPE* tmpResultsList = (DATA_TYPE*) tmpResultsMem;
	int tmpNumOfPoints;	// This need to be int because of MPI

	#ifdef DBG_TIME
		Stopwatch sw;
	#endif

	for(int i=0; i<numOfSlaves; i++){
		slaveProcessInfo[i].id = i + 1;
		slaveProcessInfo[i].currentBatchSize = parameters->batchSize;
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
				if(pinfo.jobsCompleted < parameters->slowStartLimit){
					pinfo.maxBatchSize = min(pinfo.maxBatchSize, (unsigned long) (parameters->slowStartBase * pow(2, pinfo.jobsCompleted)));
					#ifdef DBG_RATIO
						printf("[%d] Master: Setting temporary maxBatchSize=%lu for slave %d\n", rank, pinfo.maxBatchSize, mpiSource);
					#endif
				}

				// Get next data batch to calculate
				if(totalSent == totalElements){
					pinfo.work.startPoint = 0;
					pinfo.work.numOfElements = 0;
				}else{
					pinfo.work.startPoint = totalSent;
					pinfo.work.numOfElements = min(min((unsigned long) (pinfo.ratio * parameters->batchSize), (unsigned long) pinfo.maxBatchSize), totalElements-totalSent);

					//printf("pinfo.ratio = %f, paramters->batchSize = %ld, pinfo.maxBatchSize = %ld, totalElements = %ld, totalSent = %ld, product = %lu\n",
					//pinfo.ratio, parameters->batchSize, pinfo.maxBatchSize, totalElements, totalSent, (unsigned long) (pinfo.ratio * parameters->batchSize));
					totalSent += pinfo.work.numOfElements;
				}

				#ifdef DBG_MPI_STEPS
					printf("[%d] Master: Sending %lu elements to %d with index %lu\n", rank, pinfo.work.numOfElements, mpiSource, pinfo.work.startPoint);
				#endif
				#ifdef DBG_DATA
					printf("[%d] Master: Sending %lu elements to %d with index %lu\n", rank, pinfo.work.numOfElements, mpiSource, pinfo.work.startPoint);
				#endif

				// Send the batch to the slave process
				MPI_Send(&pinfo.work, 2, MPI_UNSIGNED_LONG, mpiSource, TAG_DATA, MPI_COMM_WORLD);

				// Start stopwatch for process
				pinfo.stopwatch.start();

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
					MMPI_Recv(tmpResults, pinfo.work.numOfElements, RESULT_MPI_TYPE, mpiSource, TAG_RESULTS_DATA, MPI_COMM_WORLD, &status);
				}else{
					MMPI_Recv(tmpResultsList, parameters->overrideMemoryRestrictions ? INT_MAX : pinfo.work.numOfElements * parameters->D, DATA_MPI_TYPE, mpiSource, TAG_RESULTS_DATA, MPI_COMM_WORLD, &status);

					// Find the number of points in list
					MPI_Get_count(&status, DATA_MPI_TYPE, &tmpNumOfPoints);

					// MPI_Get_count returned the count of DATA_TYPE elements received, so divide with D to get the count of points
					tmpNumOfPoints /= parameters->D;
				}

				this->totalReceived += pinfo.work.numOfElements;

				masterStopwatch.stop();
				t = masterStopwatch.getMsec()/1000;
				eta = t * ((float)totalElements/totalReceived) - t;

				if(parameters->printProgress){
					printf("Progress: %lu/%lu, %.2f %%", this->totalReceived, this->totalElements, ((float)this->totalReceived / this->totalElements)*100);

					if(t < 3600)	printf(", Elapsed time: %02d:%02d", t/60, t%60);
					else			printf(", Elapsed time: %02d:%02d:%02d", t/3600, (t%3600)/60, t%60);

					if(eta < 3600)	printf(", ETA: %02d:%02d\n", eta/60, eta%60);
					else			printf(", ETA: %02d:%02d:%02d\n", eta/3600, (eta%3600)/60, eta%60);
				}

				#ifdef DBG_MPI_STEPS
					if(parameters->resultSaveType == SAVE_TYPE_ALL)
						printf("[%d] Master: Saving %lu results from slave %d to finalResults[%lu]...\n", rank, pinfo.work.numOfElements, mpiSource, pinfo.work.startPoint);
					else
						printf("[%d] Master: Saving %lu points from slave %d to listResults[%lu]...\n", rank, tmpNumOfPoints, mpiSource, listResultsSaved);

				#endif
				#ifdef DBG_RESULTS
					if(parameters->resultSaveType == SAVE_TYPE_ALL){
						printf("[%d] Master: Saving tmpResults: ", rank);
						for (int i = 0; i < pinfo.work.numOfElements; i++) {
							printf("%f ", tmpResults[i]);
						}
						printf(" at %lu\n", pinfo.work.startPoint);
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
						memcpy(&finalResults[pinfo.work.startPoint], tmpResults, pinfo.work.numOfElements*sizeof(RESULT_TYPE));
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
				pinfo.elementsCalculated += pinfo.work.numOfElements;
				pinfo.stopwatch.stop();
				pinfo.lastScore = pinfo.work.numOfElements / pinfo.stopwatch.getMsec();
				pinfo.lastAssignedElements = pinfo.work.numOfElements;

				// Print benchmark results
				if (parameters->benchmark && parameters->printProgress) {
					printf("[%d] Master: Slave %d benchmark: %lu elements, %f ms\n\n", rank, mpiSource, pinfo.work.numOfElements, pinfo.stopwatch.getMsec());
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
				pinfo.work.numOfElements = 0;

				#ifdef DBG_TIME
					sw.stop();
					printf("[%d] Master: Benchmark: Time for TAG_RESULTS: %f ms\n", rank, sw.getMsec());
				#endif

				break;

			case TAG_EXITING:
				#ifdef DBG_MPI_STEPS
					printf("[%d] Master: Slave %d exiting\n", rank, mpiSource);
				#endif

				if(pinfo.work.numOfElements != 0){
					printf("[%d] Master: Error: Slave %d exited with %lu assigned elements!!\n", rank, mpiSource, pinfo.work.numOfElements);
				}

				pinfo.work.startPoint = totalElements;
				pinfo.finished = true;

				finished++;

				break;
		}
	}

	free(tmpResultsMem);
	delete[] slaveProcessInfo;
}

void ParallelFramework::coordinatorThread(ComputeThreadInfo* cti, ThreadCommonData* tcd, int numOfThreads){
	int* globalListIndexPtr = &tcd->listIndex;

	AssignedWork work;
	unsigned long maxBatchSize;
	unsigned long allocatedElements = 0;
	RESULT_TYPE* localResults = nullptr;
	MPI_Status status;
	int numOfRatioAdjustments = 0;

	#ifdef DBG_TIME
		Stopwatch sw;
		float time_data, time_assign, time_start, time_wait, time_scores, time_results;
	#endif

	if(parameters->overrideMemoryRestrictions){
		maxBatchSize = parameters->batchSize;
	}else{
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
	}


	#ifdef DBG_START_STOP
		printf("[%d] Coordinator: Max batch size: %lu\n", rank, maxBatchSize);
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
		MMPI_Recv(&work, 2, MPI_UNSIGNED_LONG, 0, TAG_DATA, MPI_COMM_WORLD, &status);

		#ifdef DBG_DATA
			printf("[%d] Coordinator: Received %lu elements starting from %lu\n", rank, work.numOfElements, work.startPoint);
		#endif
		#ifdef DBG_MPI_STEPS
			printf("[%d] Coordinator: Received %lu elements\n", rank, work.numOfElements);
		#endif

		// If no elements, break
		if(work.numOfElements == 0)
			break;

		// Make sure we have enough allocated memory for localResults
		if(work.numOfElements > allocatedElements && allocatedElements < getDefaultCPUBatchSize()){
			#ifdef DBG_MEMORY
				printf("[%d] Coordinator: Allocating more memory for localResults: %lu (0x%x) -> ", rank, allocatedElements, localResults);
			#endif

			allocatedElements = min(work.numOfElements, getDefaultCPUBatchSize());
			localResults = (RESULT_TYPE*) realloc(localResults, allocatedElements * sizeof(RESULT_TYPE));
			if(localResults == nullptr){
				fatal("Can't allocate memory for localResults");
			}

			tcd->results = localResults;

			#ifdef DBG_MEMORY
				printf("%lu (0x%x)\n", allocatedElements, localResults);
			#endif
		}

		#ifdef DBG_TIME
			sw.stop();
			time_data = sw.getMsec();
			sw.start();
		#endif

		/*
		 * Split the data into pieces for each thread
		 * If we are using dynamic scheduling, each batch size will be
		 * ratio * min(slaveBatchSize, numOfElements).
		 * Otherwise it will be ratio * work.numOfElements and we will make sure
		 * later that exactly work.numOfElements elements have been assigned.
		 */
		unsigned long total = 0;
		for(int i=0; i<numOfThreads; i++){
			if(parameters->slaveDynamicScheduling)
				cti[i].batchSize = cti[i].ratio * min(
					parameters->slaveBatchSize, work.numOfElements
				);
			else{
				cti[i].batchSize = cti[i].ratio * work.numOfElements;
				total += cti[i].batchSize;
			}
		}

		/*
		 * If we are NOT using dynamic scheduling, make sure that exactly
		 * work.numOfElements elements have been assigned
		 */
		if(!parameters->slaveDynamicScheduling){
			if(total < work.numOfElements){
				// Something was left out, assign it to first thread
				cti[0].batchSize += work.numOfElements - total;
			}else if(total > work.numOfElements){
				// More elements were given, remove from the first threads
				for(int i=0; i<numOfThreads; i++){
					// If thread i has enough elements, remove all the extra from it
					if(cti[i].batchSize > total - work.numOfElements){
						cti[i].batchSize -= total - work.numOfElements;
						total = work.numOfElements;
						break;
					}

					// else assign 0 elements to it and move to the next thread
					total -= cti[i].batchSize;
					cti[i].batchSize = 0;
				}
			}
		}

		#ifdef DBG_RATIO
			for(int i=0; i<numOfThreads; i++)
				printf("[%d] Coordinator: Thread %d -> Assigning batch size = %lu elements\n",
						rank, i, cti[i].batchSize);
		#endif

		tcd->globalFirst		= work.startPoint;
		tcd->globalLast			= work.startPoint + work.numOfElements - 1;
		tcd->globalBatchStart	= work.startPoint;

		#ifdef DBG_DATA
			printf("[%d] Coordinator: Got job with globalFirst = %lu, globalLast = %lu\n",
							rank, tcd->globalFirst, tcd->globalLast);
		#endif

		#ifdef DBG_TIME
			sw.stop();
			time_assign = sw.getMsec();
			sw.start();
		#endif

		// Reset the global listIndex counter
		*globalListIndexPtr = 0;

		// Start all the worker threads
		for(int i=0; i<numOfThreads; i++){
			cti[i].elementsCalculated = 0;
			sem_post(&cti[i].semStart);
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
			sem_wait(&tcd->semResults);
		}

		#ifdef DBG_TIME
			sw.stop();
			time_wait = sw.getMsec();
		#endif

		// Sanity check
		unsigned long totalCalculated = 0;
		for(int i=0; i<numOfThreads; i++)
			totalCalculated += cti[i].elementsCalculated;

		if(totalCalculated != work.numOfElements){
			printf("[%d] Coordinator: Shit happened. Total calculated elements are %lu but %lu were assigned to this slave\n",
								rank, totalCalculated, work.numOfElements);

			exit(-123);
		}

		#ifdef DBG_RESULTS
			if(parameters->resultSaveType == SAVE_TYPE_ALL){
				printf("[%d] Coordinator: Results: [", rank);
				for(int i=0; i<work.numOfElements; i++){
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
				printf("\n");
			}
		#endif

		if(parameters->threadBalancing){
			#ifdef DBG_TIME
				sw.start();
			#endif

			bool failed = false;
			float totalScore = 0;
			float newRatio;
			float *scores = new float[numOfThreads];

			for(int i=0; i<numOfThreads; i++){
				if(cti[i].stopwatch.getMsec() < 0){
					printf("[%d] Coordinator: Error: Skipping ratio correction due to negative time\n", rank);
					failed = true;
					break;
				}

				if(cti[i].stopwatch.getMsec() < parameters->minMsForRatioAdjustment){
					#ifdef DBG_RATIO
						printf("[%d] Coordinator: Skipping ratio correction due to fast execution\n", rank);
					#endif

					failed = true;
					break;
				}

				if(parameters->benchmark){
					printf("[%d] Coordinator: Thread %d time: %f ms\n", rank, cti[i].id, cti[i].stopwatch.getMsec());
				}

				scores[i] = cti[i].elementsCalculated / cti[i].stopwatch.getMsec();
				totalScore += scores[i];
			}

			if(!failed){
				// Adjust the ratio for each thread
				numOfRatioAdjustments++;
				for(int i=0; i<numOfThreads; i++){
					newRatio = scores[i] / totalScore;
					cti[i].totalRatio += newRatio;

					if(parameters->threadBalancingAverage){
						cti[i].ratio = cti[i].totalRatio / numOfRatioAdjustments;
					}else{
						cti[i].ratio = newRatio;
					}

					#ifdef DBG_RATIO
						printf("[%d] Coordinator: Adjusting thread %d ratio to %f (elements = %d, time = %f ms, score = %f)\n",
								rank, cti[i].id, cti[i].ratio, cti[i].elementsCalculated,
								cti[i].stopwatch.getMsec(), cti[i].elementsCalculated/cti[i].stopwatch.getMsec());
					#endif
				}
			}

			#ifdef DBG_TIME
				sw.stop();
				time_scores = sw.getMsec();
				sw.start();
			#endif

			delete[] scores;
		}

		// Send all results to master
		#ifdef DBG_MPI_STEPS
			printf("[%d] Coordinator: Sending data to master...\n", rank);
		#endif

		#ifdef DBG_TIME
			sw.start();
		#endif

		MPI_Send(nullptr, 0, MPI_INT, 0, TAG_RESULTS, MPI_COMM_WORLD);
		if(parameters->resultSaveType == SAVE_TYPE_ALL){
			// Send all the results
			MPI_Send(localResults, work.numOfElements, RESULT_MPI_TYPE, 0, TAG_RESULTS_DATA, MPI_COMM_WORLD);
		}else{
			// Send the list of points
			MPI_Send(localResults, *globalListIndexPtr, DATA_MPI_TYPE, 0, TAG_RESULTS_DATA, MPI_COMM_WORLD);
		}

		#ifdef DBG_TIME
			sw.stop();
			time_results = sw.getMsec();

			printf("[%d] Coordinator: Benchmark:\n", rank);
			printf("Time for receiving data: %f ms\n", time_data);
			printf("Time for assign: %f ms\n", time_assign);
			printf("Time for start: %f ms\n", time_start);
			printf("Time for wait: %f ms\n", time_wait);
			printf("Time for scores: %f ms\n", time_scores);
			printf("Time for results: %f ms\n", time_results);
		#endif
	}

	// Notify about exiting
	MPI_Send(nullptr, 0, MPI_INT, 0, TAG_EXITING, MPI_COMM_WORLD);

	// Signal worker threads to finish
	tcd->globalFirst = 1;
	tcd->globalLast = 0;
	for(int i=0; i<numOfThreads; i++){
		sem_post(&cti[i].semStart);
	}

	free(localResults);
}

RESULT_TYPE* ParallelFramework::getResults() {
	if(rank != 0){
		printf("Error: Results can only be fetched by the master process. Are you the master process?\n");
		return nullptr;
	}

	if(parameters->resultSaveType != SAVE_TYPE_ALL){
		printf("Error: Can't get all results when resultSaveType is not SAVE_TYPE_ALL\n");
		return nullptr;
	}

	return finalResults;
}
DATA_TYPE* ParallelFramework::getList(int* length){
	if(rank != 0){
		printf("Error: Results can only be fetched by the master process. Are you the master process?\n");
		return nullptr;
	}

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

int ParallelFramework::getRank(){
	return this->rank;
}
