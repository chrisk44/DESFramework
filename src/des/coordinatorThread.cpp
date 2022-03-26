#include "framework.h"

void ParallelFramework::coordinatorThread(ComputeThreadInfo* cti, ThreadCommonData* tcd, int numOfThreads){
	int* globalListIndexPtr = &tcd->listIndex;

	AssignedWork work;
	unsigned long maxBatchSize;
	unsigned long allocatedElements = 0;
	RESULT_TYPE* localResults = nullptr;
	MPI_Status status;
	int numOfRatioAdjustments = 0;
	unsigned long maxCpuElements = getMaxCPUBytes();
    if(parameters.resultSaveType == SAVE_TYPE_ALL)
		maxCpuElements /= sizeof(RESULT_TYPE);
	else
        maxCpuElements /= parameters.D * sizeof(DATA_TYPE);

	#ifdef DBG_TIME
		Stopwatch sw;
		float time_data, time_assign, time_start, time_wait, time_scores, time_results;
	#endif

    if(parameters.overrideMemoryRestrictions){
        maxBatchSize = parameters.batchSize;
	}else{
        if(parameters.processingType == PROCESSING_TYPE_CPU)
			maxBatchSize = getMaxCPUBytes();
        else if(parameters.processingType == PROCESSING_TYPE_GPU)
			maxBatchSize = getMaxGPUBytes();
		else
            maxBatchSize = std::min(getMaxCPUBytes(), getMaxGPUBytes());

		// maxBatchSize contains the value in bytes, so divide it according to resultSaveType to convert it to actual batch size
        if(parameters.resultSaveType == SAVE_TYPE_ALL)
            maxBatchSize /= sizeof(RESULT_TYPE);
        else
            maxBatchSize /= parameters.D * sizeof(DATA_TYPE);

		// Limit the batch size by the user-given value
        maxBatchSize = std::min((unsigned long)parameters.batchSize, (unsigned long)maxBatchSize);

		// If we are saving a list, the max number of elements we might want to send is maxBatchSize * D, so limit the batch size
		// so that the max number of elements is INT_MAX
        if(parameters.resultSaveType == SAVE_TYPE_LIST && (unsigned long) (maxBatchSize*parameters.D) > (unsigned long) INT_MAX){
            maxBatchSize = (INT_MAX - parameters.D) / parameters.D;
		}
		// If we are saving all of the results, the max number of elements is maxBatchSize itself, so limit it to INT_MAX
        else if(parameters.resultSaveType == SAVE_TYPE_ALL && (unsigned long) maxBatchSize > (unsigned long) INT_MAX){
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
		if(work.numOfElements > allocatedElements && allocatedElements < maxCpuElements){
			#ifdef DBG_MEMORY
				printf("[%d] Coordinator: Allocating more memory for localResults: %lu (0x%x) -> ", rank, allocatedElements, localResults);
			#endif

            allocatedElements = std::min(work.numOfElements, maxCpuElements);
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
            if(parameters.slaveDynamicScheduling)
                cti[i].batchSize = std::max((unsigned long) 1, (unsigned long) (cti[i].ratio * std::min(
                    parameters.slaveBatchSize, work.numOfElements)));
			else{
                cti[i].batchSize = std::max((unsigned long) 1, (unsigned long) (cti[i].ratio * work.numOfElements));
				total += cti[i].batchSize;
			}
		}

		/*
		 * If we are NOT using dynamic scheduling, make sure that exactly
		 * work.numOfElements elements have been assigned
		 */
        if(!parameters.slaveDynamicScheduling){
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
            printf("[%d] Coordinator: Waiting for results from %d threads...\n", rank, numOfThreads);
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
            if(parameters.resultSaveType == SAVE_TYPE_ALL){
				printf("[%d] Coordinator: Results: [", rank);
                for(unsigned long i=0; i<work.numOfElements; i++){
					printf("%f ", localResults[i]);
				}
				printf("]\n");
			}else{
				printf("[%d] Coordinator: Results (*globalListIndexPtr = %d):", rank, *globalListIndexPtr);
                for(int i=0; i<*globalListIndexPtr; i+=parameters.D){
					printf("[ ");
                    for(unsigned int j=0; j<parameters.D; j++){
						printf("%f ", ((DATA_TYPE *)localResults)[i + j]);
					}
					printf("]");
				}
				printf("\n");
			}
		#endif

        if(parameters.threadBalancing){
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

                if(cti[i].stopwatch.getMsec() < parameters.minMsForRatioAdjustment){
					#ifdef DBG_RATIO
						printf("[%d] Coordinator: Skipping ratio correction due to fast execution\n", rank);
					#endif

					failed = true;
					break;
				}

                if(parameters.benchmark){
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

                    if(parameters.threadBalancingAverage){
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
        if(parameters.resultSaveType == SAVE_TYPE_ALL){
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

