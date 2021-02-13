#ifndef PARALLELFRAMEWORK_H
#define PARALLELFRAMEWORK_H

#include <cuda.h>
#include <nvml.h>

#include <cmath>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <unistd.h>

#include "utilities.h"
#include "kernels.cu"

using namespace std;

class ParallelFramework {
private:
	// Parameters
	Limit* limits = NULL;						// This must be an array of length = parameters.D
	ParallelFrameworkParameters* parameters = NULL;

	// Runtime variables
	unsigned long long* idxSteps = NULL;		// Index steps for each dimension
	int saveFile = -1;							// File descriptor for save file
	RESULT_TYPE* finalResults = NULL;			// An array of N0 * N1 * ... * N(D-1)
	DATA_TYPE* listResults = NULL;				// An array of points for which the validation function has returned non-zero value
	int listResultsSaved = 0;					// Number of points saved in listResults
	bool valid = false;
	unsigned long totalSent = 0;			// Total elements that have been sent for processing, also the index from which the next assigned batch will start
	unsigned long totalReceived = 0;		// TOtal elements that have been calculated and returned
	unsigned long totalElements = 0;		// Total elements

	// MPI
	int rank = -1;

public:
	ParallelFramework(bool initMPI);
	~ParallelFramework();
	void init(Limit* limits, ParallelFrameworkParameters& parameters);

	template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
	int run();

	RESULT_TYPE* getResults();
	DATA_TYPE* getList(int* length);
	void getIndicesFromPoint(DATA_TYPE* point, unsigned long* dst);
	unsigned long getIndexFromIndices(unsigned long* pointIdx);
	unsigned long getIndexFromPoint(DATA_TYPE* point);
	bool isValid();
	int getRank();

private:
	void masterProcess();
	void coordinatorThread(ComputeThreadInfo* cti, ThreadCommonData* tcd, int numOfThreads);
	void getPointFromIndex(unsigned long index, DATA_TYPE* result);

	template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
	void slaveProcess();

	template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
	void computeThread(ComputeThreadInfo& cti, ThreadCommonData* tcd);
};

template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
int ParallelFramework::run() {
	if(!valid){
		printf("[%d] run() called for invalid framework\n", rank);
		return -1;
	}

	if(rank == 0){

		if(parameters->printProgress)
			printf("[%d] Master process starting\n", rank);

		masterProcess();

		if(parameters->printProgress)
			printf("[%d] Master process finished\n", rank);

	}else{

		if(parameters->printProgress)
			printf("[%d] Slave process starting\n", rank);

		slaveProcess<validation_cpu, validation_gpu, toBool_cpu, toBool_gpu>();

		if(parameters->printProgress)
			printf("[%d] Slave process finished\n", rank);

	}

	if(parameters->finalizeAfterExecution)
		MPI_Finalize();

	return 0;
}

template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
void ParallelFramework::slaveProcess() {
	/*******************************************************************
	********** Calculate number of worker threads (#GPUs + 1CPU) *******
	********************************************************************/
	int numOfThreads = 0;

	if(parameters->processingType != PROCESSING_TYPE_CPU)
		cudaGetDeviceCount(&numOfThreads);

	if(parameters->processingType != PROCESSING_TYPE_GPU)
		numOfThreads++;

	if(numOfThreads == 0){
		printf("[%d] SlaveProcess: Error: cudaGetDeviceCount returned 0\n", rank);
		return;
	}


	/*******************************************************************
	********************** Initialization ******************************
	********************************************************************/
	ThreadCommonData threadCommonData;
	sem_init(&threadCommonData.semResults, 0, 0);
	sem_init(&threadCommonData.semSync, 0, 1);
	threadCommonData.results = nullptr;

	ComputeThreadInfo* computeThreadInfo = new ComputeThreadInfo[numOfThreads];
	for(int i=0; i<numOfThreads; i++){
		sem_init(&computeThreadInfo[i].semStart, 0, 0);
		computeThreadInfo[i].ratio = (float)1/numOfThreads;
		computeThreadInfo[i].totalRatio = 0;
	}

	#ifdef DBG_START_STOP
		printf("[%d] SlaveProcess: Spawning %d worker threads...\n", rank, numOfThreads);
	#endif

	/*******************************************************************
	*************** Launch coordinator and worker threads **************
	********************************************************************/
	#pragma omp parallel num_threads(numOfThreads + 1) shared(computeThreadInfo) 	// +1 thread to handle the communication with masterProcess
	{
		int tid = omp_get_thread_num();
		if(tid == 0){
			coordinatorThread(computeThreadInfo, &threadCommonData, omp_get_num_threads()-1);
		}else{
			// Calculate id: -1 -> CPU, 0+ -> GPU[id]
			computeThreadInfo[tid-1].id = tid - (parameters->processingType == PROCESSING_TYPE_GPU ? 1 : 2);

			computeThread<validation_cpu, validation_gpu, toBool_cpu, toBool_gpu>(computeThreadInfo[tid - 1], &threadCommonData);
		}
	}

	/*******************************************************************
	***************************** Finalize *****************************
	********************************************************************/
	sem_destroy(&threadCommonData.semResults);
	sem_destroy(&threadCommonData.semSync);
	for(int i=0; i<numOfThreads; i++){
		sem_destroy(&computeThreadInfo[i].semStart);
	}
	delete[] computeThreadInfo;
}

template<validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu>
void ParallelFramework::computeThread(ComputeThreadInfo& cti, ThreadCommonData* tcd){
	int gpuListIndex, globalListIndexOld;
	RESULT_TYPE* localResults;

	// GPU Memory
	RESULT_TYPE* deviceResults;					// GPU Memory for results
	int* deviceListIndexPtr;					// GPU Memory for list index for synchronization when saving the results as a list of points
	void* deviceDataPtr;						// GPU Memory to store the model's constant data

	// GPU Runtime
	cudaStream_t streams[parameters->gpuStreams];
	unsigned long allocatedElements = 0;
	cudaDeviceProp deviceProp;
	unsigned long defaultGPUBatchSize = getDefaultGPUBatchSize();
	bool useSharedMemoryForData;
	bool useConstantMemoryForData;
	int maxSharedPoints;
	int availableSharedMemory;

	// NVML
	bool nvmlInitialized = false;
	bool nvmlAvailable = false;
	nvmlReturn_t result;
	nvmlDevice_t gpuHandle;
	char gpuName[NVML_DEVICE_NAME_BUFFER_SIZE];

	float totalUtilization = 0;
	unsigned long numOfSamples = 0;

	nvmlSample_t *samples = new nvmlSample_t[1];
	unsigned long numOfAllocatedSamples = 1;
	unsigned long long lastSeenTimeStamp = 0;
	nvmlValueType_t sampleValType;

	/*******************************************************************
	************************* Initialization ***************************
	********************************************************************/
	// Initialize device
	if (cti.id > -1) {
		// Select gpu[id]
		cudaSetDevice(cti.id);

		// Get device's properties for shared memory
		cudaGetDeviceProperties(&deviceProp, cti.id);

		// Use constant memory for data if they fit
		useConstantMemoryForData = parameters->dataSize > 0 &&
			parameters->dataSize <= (MAX_CONSTANT_MEMORY - parameters->D * (sizeof(Limit) + sizeof(unsigned long long)));

		// Max use 1/4 of the available shared memory for data, the rest will be used for each thread to store their point (x) and index vector (i)
		// This seems to be worse than both global and constant memory
		useSharedMemoryForData = false && parameters->dataSize > 0 && !useConstantMemoryForData &&
								 parameters->dataSize <= deviceProp.sharedMemPerBlock / 4;

		// How many bytes are left in shared memory after using it for the model's data
		availableSharedMemory = deviceProp.sharedMemPerBlock - (useSharedMemoryForData ? parameters->dataSize : 0);

		// How many points can fit in shared memory (for each point we need D*DATA_TYPEs (for x) and D*u_int (for indices))
		maxSharedPoints = availableSharedMemory / (parameters->D * (sizeof(DATA_TYPE) + sizeof(unsigned int)));

		#ifdef DBG_START_STOP
			if(parameters->printProgress){
				printf("[%d] ComputeThread %d: useSharedMemoryForData = %d\n", rank, cti.id, useSharedMemoryForData);
				printf("[%d] ComputeThread %d: useConstantMemoryForData = %d\n", rank, cti.id, useConstantMemoryForData);
				printf("[%d] ComputeThread %d: availableSharedMemory = %d bytes\n", rank, cti.id, availableSharedMemory);
				printf("[%d] ComputeThread %d: maxSharedPoints = %d\n", rank, cti.id, maxSharedPoints);
			}
		#endif

		// Create streams
		for(int i=0; i<parameters->gpuStreams; i++){
			cudaStreamCreate(&streams[i]);
			cce();
		}

		// Allocate memory on device
		cudaMalloc(&deviceResults, allocatedElements * sizeof(RESULT_TYPE));	cce();
		cudaMalloc(&deviceListIndexPtr, sizeof(int));							cce();
		// If we have data but can't fit it in constant memory, allocate global memory
		if(parameters->dataSize > 0 && !useConstantMemoryForData){
			cudaMalloc(&deviceDataPtr, parameters->dataSize);					cce();
		}

		#ifdef DBG_MEMORY
			printf("[%d] ComputeThread %d: deviceResults: 0x%x\n", rank, cti.id, (void*) deviceResults);
			printf("[%d] ComputeThread %d: deviceListIndexPtr: 0x%x\n", rank, cti.id, (void*) deviceListIndexPtr);
			printf("[%d] ComputeThread %d: deviceDataPtr: 0x%x\n", rank, cti.id, (void*) deviceDataPtr);
		#endif

		// Copy limits, idxSteps, and constant data to device
		#ifdef DBG_MEMORY
			printf("[%d] ComputeThread %d: Copying limits at constant memory with offset %d\n",
											rank, cti.id, 0);
			printf("[%d] ComputeThread %d: Copying idxSteps at constant memory with offset %d\n",
											rank, cti.id, parameters->D * sizeof(Limit));
		#endif
		cudaMemcpyToSymbolWrapper<nullptr, nullptr>(
			limits, parameters->D * sizeof(Limit), 0);
		cce();

		cudaMemcpyToSymbolWrapper<nullptr, nullptr>(
			idxSteps, parameters->D * sizeof(unsigned long long),
			parameters->D * sizeof(Limit));
		cce();

		// If we have data for the model...
		if(parameters->dataSize > 0){
			// If we can use constant memory, copy it there
			if(useConstantMemoryForData){
				#ifdef DBG_MEMORY
					printf("[%d] ComputeThread %d: Copying data at constant memory with offset %d\n",
										rank, cti.id, parameters->D * (sizeof(Limit) + sizeof(unsigned long long)));
				#endif
				cudaMemcpyToSymbolWrapper<nullptr, nullptr>(
					parameters->dataPtr, parameters->dataSize,
					parameters->D * (sizeof(Limit) + sizeof(unsigned long long)));
				cce()
			}
			// else copy the data to the global memory, either to be read from there or to be copied to shared memory
			else{
				cudaMemcpy(deviceDataPtr, parameters->dataPtr, parameters->dataSize, cudaMemcpyHostToDevice);
				cce();
			}
		}

		// Initialize NVML to monitor the GPU
		nvmlAvailable = false;
		result = nvmlInit();
	    if (result == NVML_SUCCESS){
			nvmlInitialized = true;

			result = nvmlDeviceGetHandleByIndex(cti.id, &gpuHandle);
			if(result == NVML_SUCCESS){
				nvmlAvailable = true;
			}else{
				printf("[%d] [E] Failed to get device handle for gpu %d: %s\n", rank, cti.id, nvmlErrorString(result));
			}
		} else {
	        printf("[%d] [E] Failed to initialize NVML: %s\n", rank, nvmlErrorString(result));
			nvmlAvailable = false;
	    }

		if(nvmlAvailable){
			// Get the device's name for later
			result = nvmlDeviceGetName(gpuHandle, gpuName, NVML_DEVICE_NAME_BUFFER_SIZE);

			// Get samples to save the current timestamp
			unsigned int temp = 1;
			// Read them one by one to avoid allocating memory for the whole buffer
			while((result = nvmlDeviceGetSamples(gpuHandle, NVML_GPU_UTILIZATION_SAMPLES, lastSeenTimeStamp, &sampleValType, &temp, samples)) == NVML_SUCCESS && temp > 0){
				lastSeenTimeStamp = samples[temp-1].timeStamp;
			}

			if (result != NVML_SUCCESS && result != NVML_ERROR_NOT_FOUND) {
				printf("[%d] [E] Failed to get initial utilization samples for device: %s\n", rank, nvmlErrorString(result));
				nvmlAvailable = false;
			}
		}
	}


	/*******************************************************************
	************************* Execution loop ***************************
	********************************************************************/
	while(true){
		#ifdef DBG_TIME
			Stopwatch sw;
			float time_sleep, time_assign=0, time_allocation=0, time_queue=0, time_calc=0;
			sw.start();
		#endif

		/*****************************************************************
		************* Wait for data from coordinateThread ****************
		******************************************************************/
		#ifdef DBG_START_STOP
			printf("[%d] ComputeThread %d: Ready\n", rank, cti.id);
		#endif

		sem_wait(&cti.semStart);

		#ifdef DBG_TIME
			sw.stop();
			time_sleep = sw.getMsec();
		#endif

		cti.stopwatch.start();

		#ifdef DBG_START_STOP
			printf("[%d] ComputeThread %d: Waking up...\n", rank, cti.id);
		#endif

		// If coordinator thread is signaling to terminate...
		if (tcd->globalFirst > tcd->globalLast)
			break;

		/*****************************************************************
		 ***************** Main batch assignment loop ********************
		 ******* (will run only once if !slaveDynamicScheduling) *********
		 *****************************************************************/
		while(cti.batchSize > 0){	// No point in running with batch size = 0

			unsigned long localStartPoint;
			unsigned long localLast;
			unsigned long localNumOfElements;

			/*****************************************************************
			************************** Get a batch ***************************
			******************************************************************/
			#ifdef DBG_TIME
				sw.start();
			#endif

			sem_wait(&tcd->semSync);

			// Get the current global batch start point as our starting point
			localStartPoint = tcd->globalBatchStart;
			// Increment the global batch start point by our batch size
			tcd->globalBatchStart += cti.batchSize;

			// Check for globalBatchStart overflow and limit it to globalLast+1 to avoid later overflows
			// If the new globalBatchStart is smaller than our local start point, the increment caused an overflow
			// If the localStart point in larger than the global last, then the elements have already been exhausted
			if(tcd->globalBatchStart < localStartPoint || localStartPoint > tcd->globalLast){
				// printf("[%d] ComputeThread %d: Fixing globalBatchStart from %lu to %lu\n",
				// 				rank, cti.id, tcd->globalBatchStart, tcd->globalLast + 1);
				tcd->globalBatchStart = tcd->globalLast + 1;
			}

			sem_post(&tcd->semSync);

			// If we are out of elements then terminate the loop
			if(localStartPoint > tcd->globalLast)
				break;

			localLast = min(localStartPoint + cti.batchSize - 1 , tcd->globalLast);
			localNumOfElements = localLast - localStartPoint + 1;

			if(parameters->resultSaveType == SAVE_TYPE_LIST)
				localResults = tcd->results;
			else
				localResults = &tcd->results[localStartPoint - tcd->globalFirst];

			#ifdef DBG_TIME
				sw.stop();
				time_assign += sw.getMsec();
			#endif

			#ifdef DBG_DATA
				printf("[%d] ComputeThread %d: Got %lu elements starting from %lu to %lu\n",
							rank, cti.id, localNumOfElements, localStartPoint, localLast);
				fflush(stdout);
			#else
				#ifdef DBG_START_STOP
					printf("[%d] ComputeThread %d: Running for %lu elements...\n", rank, cti.id, localNumOfElements);
					fflush(stdout);
				#endif
			#endif

			#ifdef DBG_TIME
				sw.start();
			#endif

			/*****************************************************************
			 If batchSize was increased, allocate more memory for the results
			******************************************************************/
			// TODO: Move this to initialization and allocate as much memory as possible
			if (allocatedElements < localNumOfElements && cti.id > -1 && allocatedElements < defaultGPUBatchSize) {

				#ifdef DBG_MEMORY
					printf("[%d] ComputeThread %d: Allocating more GPU memory (%lu", rank, cti.id, allocatedElements);
					fflush(stdout);
				#endif

				allocatedElements = min(localNumOfElements, defaultGPUBatchSize);

				#ifdef DBG_MEMORY
					printf(" -> %lu elements, %lu MB)\n", allocatedElements, (allocatedElements*sizeof(RESULT_TYPE)) / (1024 * 1024));
					fflush(stdout);
				#endif

				// Reallocate memory on device
				cudaFree(deviceResults);
				cce();
				cudaMalloc(&deviceResults, allocatedElements * sizeof(RESULT_TYPE));
				cce();

				#ifdef DBG_MEMORY
					printf("[%d] ComputeThread %d: deviceResults = 0x%x\n", rank, cti.id, deviceResults);
				#endif
			}

			#ifdef DBG_TIME
				sw.stop();
				time_allocation += sw.getMsec();
				sw.start();
			#endif

			/*****************************************************************
			******************** Calculate the results ***********************
			******************************************************************/
			// If GPU...
			if (cti.id > -1) {

				// Initialize the list index counter
				cudaMemset(deviceListIndexPtr, 0, sizeof(int));

				// Divide the chunk to smaller chunks to scatter accross streams
				unsigned long elementsPerStream = localNumOfElements / parameters->gpuStreams;
				bool onlyOne = false;
				unsigned long skip = 0;
				if(elementsPerStream == 0){
					elementsPerStream = localNumOfElements;
					onlyOne = true;
				}

				// Queue the chunks to the streams
				for(int i=0; i<parameters->gpuStreams; i++){
					// Adjust elementsPerStream for last stream (= total-queued)
					if(i == parameters->gpuStreams - 1){
						elementsPerStream = localNumOfElements - skip;
					}else{
						elementsPerStream = min(elementsPerStream, localNumOfElements - skip);
					}

					// Queue the kernel in stream[i] (each GPU thread gets COMPUTE_BATCH_SIZE elements to calculate)
					int gpuThreads = (elementsPerStream + parameters->computeBatchSize - 1) / parameters->computeBatchSize;

					// Minimum of (minimum of user-defined block size and number of threads to go to this stream) and number of points that can fit in shared memory
					int blockSize = min(min(parameters->blockSize, gpuThreads), maxSharedPoints);
					int numOfBlocks = (gpuThreads + blockSize - 1) / blockSize;

					#ifdef DBG_QUEUE
						printf("[%d] ComputeThread %d: Queueing %lu elements in stream %d (%d gpuThreads, %d blocks, %d block size), with skip=%lu\n", rank,
								cti.id, elementsPerStream, i, gpuThreads, numOfBlocks, blockSize, skip);
					#endif

					// Note: Point at the start of deviceResults, because the offset (because of computeBatchSize) is calculated in the kernel
					validate_kernel<validation_gpu, toBool_gpu><<<numOfBlocks, blockSize, deviceProp.sharedMemPerBlock, streams[i]>>>(
						deviceResults, localStartPoint,
						parameters->D, elementsPerStream, skip, deviceDataPtr,
						parameters->dataSize, useSharedMemoryForData, useConstantMemoryForData,
						parameters->resultSaveType == SAVE_TYPE_ALL ? nullptr : deviceListIndexPtr,
						parameters->computeBatchSize
					);

					// Queue the memcpy in stream[i] only if we are saving as SAVE_TYPE_ALL (otherwise the results will be fetched at the end of the current computation)
					if(parameters->resultSaveType == SAVE_TYPE_ALL){
						cudaMemcpyAsync(&localResults[skip], &deviceResults[skip], elementsPerStream*sizeof(RESULT_TYPE), cudaMemcpyDeviceToHost, streams[i]);
					}

					// Increase skip
					skip += elementsPerStream;

					if(onlyOne)
						break;
				}

				#ifdef DBG_TIME
					sw.stop();
					time_queue += sw.getMsec();
					sw.start();
				#endif

				// Wait for all streams to finish
				for(int i=0; i<parameters->gpuStreams; i++){
					cudaStreamSynchronize(streams[i]);
					cce();
				}

				// If we are saving as SAVE_TYPE_LIST, fetch the results
				if(parameters->resultSaveType == SAVE_TYPE_LIST){
					// Get the current list index from the GPU
					cudaMemcpy(&gpuListIndex, deviceListIndexPtr, sizeof(int), cudaMemcpyDeviceToHost);

					// Increment the global list index counter
					globalListIndexOld = __sync_fetch_and_add(&tcd->listIndex, gpuListIndex);

					// Get the results from the GPU
					cudaMemcpy(&((DATA_TYPE*)localResults)[globalListIndexOld], deviceResults, gpuListIndex * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
				}

				if(nvmlAvailable){
					unsigned int tmpSamples;

					// Get number of available samples
					result = nvmlDeviceGetSamples(gpuHandle, NVML_GPU_UTILIZATION_SAMPLES, lastSeenTimeStamp, &sampleValType, &tmpSamples, NULL);
					if (result != NVML_SUCCESS && result != NVML_ERROR_NOT_FOUND) {
						printf("[%d] [E1] Failed to get utilization samples for device: %s\n", rank, nvmlErrorString(result));
					}else if(result == NVML_SUCCESS){

						// Make sure we have enough allocated memory for the new samples
						if(tmpSamples > numOfAllocatedSamples){
							delete[] samples;
							samples = new nvmlSample_t[tmpSamples];
						}

						result = nvmlDeviceGetSamples(gpuHandle, NVML_GPU_UTILIZATION_SAMPLES, lastSeenTimeStamp, &sampleValType, &tmpSamples, samples);
						if (result == NVML_SUCCESS) {
							numOfSamples += tmpSamples;
							for(int i=0; i<tmpSamples; i++){
								totalUtilization += samples[i].sampleValue.uiVal;
							}
						}else if(result != NVML_ERROR_NOT_FOUND){
							printf("[%d] [E2] Failed to get utilization samples for device: %s\n", rank, nvmlErrorString(result));
						}
					}

				}

			} else {

				#ifdef DBG_TIME
					sw.stop();
					time_queue += sw.getMsec();
					sw.start();
				#endif

				cpu_kernel<validation_cpu, toBool_cpu>(localResults, limits, parameters->D, localNumOfElements, parameters->dataPtr, parameters->resultSaveType == SAVE_TYPE_ALL ? nullptr : &tcd->listIndex,
							idxSteps, localStartPoint, parameters->cpuDynamicScheduling, parameters->cpuComputeBatchSize);

			}

			#ifdef DBG_TIME
				sw.stop();
				time_calc += sw.getMsec();
			#endif

			#ifdef DBG_RESULTS
				if(parameters->resultSaveType == SAVE_TYPE_ALL){
					printf("[%d] ComputeThread %d: Results are: ", rank, cti.id);
					for (int i = 0; i < localNumOfElements; i++) {
						printf("%f ", ((DATA_TYPE *)localResults)[i]);
					}
					printf("\n");
				}
			#endif

			#ifdef DBG_START_STOP
				printf("[%d] ComputeThread %d: Finished calculation\n", rank, cti.id);
			#endif

			cti.elementsCalculated += localNumOfElements;

			if(!parameters->slaveDynamicScheduling)
				break;

			// End of assignment loop
		}

		// Stop the stopwatch
		cti.stopwatch.stop();

		#ifdef DBG_TIME
			printf("[%d] ComputeThread %d: Benchmark:\n", rank, cti.id);
			printf("Time for first sleep: %f ms\n", time_sleep);
			printf("Time for assignments: %f ms\n", time_assign);
			printf("Time for allocations: %f ms\n", time_allocation);
			printf("Time for queues: %f ms\n", time_queue);
			printf("Time for calcs: %f ms\n", time_calc);
		#endif

		// Let coordinatorThread know that the results are ready
		sem_post(&tcd->semResults);
	}

	#ifdef DBG_START_STOP
		printf("[%d] ComputeThread %d: Finalizing and exiting...\n", rank, cti.id);
	#endif

	if(nvmlAvailable){
		printf("[%d] %s utilization: %.02f%% from %lu samples\n", rank, gpuName,
				numOfSamples > 0 ? totalUtilization/numOfSamples : 0, numOfSamples);
	}

	/*******************************************************************
	 *************************** Finalize ******************************
	 *******************************************************************/
	// Finalize GPU
	if (cti.id > -1) {
		// Make sure streams are finished and destroy them
		for(int i=0;i<parameters->gpuStreams;i++){
			cudaStreamDestroy(streams[i]);
			cce();
		}

		// Deallocate device's memory
		cudaFree(deviceResults);			cce();
		cudaFree(deviceListIndexPtr);		cce();
		cudaFree(deviceDataPtr);			cce();

		delete[] samples;
		if(nvmlInitialized){
			result = nvmlShutdown();
		    if (result != NVML_SUCCESS)
		        printf("[%d] [E] Failed to shutdown NVML: %s\n", rank, nvmlErrorString(result));
		}
	}
}

#endif
