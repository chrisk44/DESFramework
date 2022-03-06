#include "framework.h"

#include <cstring>

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

    SlaveProcessInfo slaveProcessInfo[numOfSlaves];
	#define pinfo (slaveProcessInfo[mpiSource-1])

	void* tmpResultsMem;
    if(parameters.overrideMemoryRestrictions){
		tmpResultsMem = malloc(getMaxCPUBytes());
	}else{
        tmpResultsMem = malloc(parameters.resultSaveType == SAVE_TYPE_ALL ? parameters.batchSize * sizeof(RESULT_TYPE) : parameters.batchSize * parameters.D * sizeof(DATA_TYPE));
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
        slaveProcessInfo[i].currentBatchSize = parameters.batchSize;
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
                if(pinfo.jobsCompleted < parameters.slowStartLimit){
                    pinfo.maxBatchSize = std::min(pinfo.maxBatchSize, (unsigned long) (parameters.slowStartBase * pow(2, pinfo.jobsCompleted)));
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
                    pinfo.work.numOfElements = std::min(std::min((unsigned long) (pinfo.ratio * parameters.batchSize), (unsigned long) pinfo.maxBatchSize), totalElements-totalSent);

					// printf("pinfo.ratio = %f, paramters->batchSize = %lu, pinfo.maxBatchSize = %lu, totalElements = %lu, totalSent = %lu, product = %lu\n",
                    // 		pinfo.ratio, parameters.batchSize, pinfo.maxBatchSize, totalElements, totalSent, (unsigned long) (pinfo.ratio * parameters.batchSize));

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
                if(parameters.resultSaveType == SAVE_TYPE_ALL){
					MMPI_Recv(tmpResults, pinfo.work.numOfElements, RESULT_MPI_TYPE, mpiSource, TAG_RESULTS_DATA, MPI_COMM_WORLD, &status);
				}else{
                    MMPI_Recv(tmpResultsList, parameters.overrideMemoryRestrictions ? INT_MAX : pinfo.work.numOfElements * parameters.D, DATA_MPI_TYPE, mpiSource, TAG_RESULTS_DATA, MPI_COMM_WORLD, &status);

					// Find the number of points in list
					MPI_Get_count(&status, DATA_MPI_TYPE, &tmpNumOfPoints);

					// MPI_Get_count returned the count of DATA_TYPE elements received, so divide with D to get the count of points
                    tmpNumOfPoints /= parameters.D;
				}

				this->totalReceived += pinfo.work.numOfElements;

				masterStopwatch.stop();
				t = masterStopwatch.getMsec()/1000;
				eta = t * ((float)totalElements/totalReceived) - t;

                if(parameters.printProgress){
					printf("Progress: %lu/%lu, %.2f %%", this->totalReceived, this->totalElements, ((float)this->totalReceived / this->totalElements)*100);

					if(t < 3600)	printf(", Elapsed time: %02d:%02d", t/60, t%60);
					else			printf(", Elapsed time: %02d:%02d:%02d", t/3600, (t%3600)/60, t%60);

					if(eta < 3600)	printf(", ETA: %02d:%02d\n", eta/60, eta%60);
					else			printf(", ETA: %02d:%02d:%02d\n", eta/3600, (eta%3600)/60, eta%60);
				}

				#ifdef DBG_MPI_STEPS
                    if(parameters.resultSaveType == SAVE_TYPE_ALL)
						printf("[%d] Master: Saving %lu results from slave %d to finalResults[%lu]...\n", rank, pinfo.work.numOfElements, mpiSource, pinfo.work.startPoint);
					else
						printf("[%d] Master: Saving %lu points from slave %d to listResults[%lu]...\n", rank, tmpNumOfPoints, mpiSource, listResultsSaved);

				#endif
				#ifdef DBG_RESULTS
                    if(parameters.resultSaveType == SAVE_TYPE_ALL){
						printf("[%d] Master: Saving tmpResults: ", rank);
						for (int i = 0; i < pinfo.work.numOfElements; i++) {
							printf("%f ", tmpResults[i]);
						}
						printf(" at %lu\n", pinfo.work.startPoint);
					}else{
						printf("[%d] Master: Saving tmpResultsList: ", rank);
						for (int i = 0; i < tmpNumOfPoints; i++){
							printf("[ ");
                            for(int j=0; j<parameters.D; j++){
                                printf("%f ", tmpResultsList[i*parameters.D + j]);
							}
							printf("]");
						}
						printf(" at %d\n", listResultsSaved);
					}
				#endif

				// Copy the received results to finalResults or listResults
                if( ! (parameters.benchmark)){
                    if(parameters.resultSaveType == SAVE_TYPE_ALL){
						memcpy(&finalResults[pinfo.work.startPoint], tmpResults, pinfo.work.numOfElements*sizeof(RESULT_TYPE));
					}else if(tmpNumOfPoints > 0){
						// Reallocate listResults
						listResultsSaved += tmpNumOfPoints;
                        listResults = (DATA_TYPE*) realloc(listResults, listResultsSaved * parameters.D * sizeof(DATA_TYPE));
						if(listResults == nullptr){
							fatal("Can't allocate memory for listResults");
						}

						// Append the received data in listResults
                        memcpy(&listResults[(listResultsSaved - tmpNumOfPoints) * parameters.D], tmpResultsList, tmpNumOfPoints * parameters.D * sizeof(DATA_TYPE));
					}
				}

				// Update pinfo
				pinfo.jobsCompleted++;
				pinfo.elementsCalculated += pinfo.work.numOfElements;
				pinfo.stopwatch.stop();
				pinfo.lastScore = pinfo.work.numOfElements / pinfo.stopwatch.getMsec();
				pinfo.lastAssignedElements = pinfo.work.numOfElements;

				// Print benchmark results
                if (parameters.benchmark && parameters.printProgress) {
					printf("[%d] Master: Slave %d benchmark: %lu elements, %f ms\n\n", rank, mpiSource, pinfo.work.numOfElements, pinfo.stopwatch.getMsec());
				}

                if(parameters.slaveBalancing && numOfSlaves > 1){
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

	// Synchronize with the rest of the processes
	#ifdef DBG_START_STOP
		printf("[%d] Waiting in barrier...\n", rank);
	#endif
	// MPI_Barrier(MPI_COMM_WORLD);
	int a = 0;
	MPI_Bcast(&a, 1, MPI_INT, 0, MPI_COMM_WORLD);
	#ifdef DBG_START_STOP
		printf("[%d] Passed the barrier...\n", rank);
	#endif
}

