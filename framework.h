#ifndef PARALLELFRAMEWORK_H
#define PARALLELFRAMEWORK_H

#include <cuda.h>
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
	unsigned long listResultsSaved = 0;			// Number of points saved in listResults
	bool valid = false;
	unsigned long long totalSent = 0;			// Total elements that have been sent for processing, also the index from which the next assigned batch will start
	unsigned long long totalReceived = 0;		// TOtal elements that have been calculated and returned
	unsigned long long totalElements = 0;		// Total elements

	// MPI
	int rank = -1;

public:
	ParallelFramework();
	~ParallelFramework();
	void init(Limit* limits, ParallelFrameworkParameters& parameters);

	template<class ImplementedModel>
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
	void coordinatorThread(ComputeThreadInfo* cti, int numOfThreads, Model* model);
	unsigned long getDataChunk(unsigned long maxBatchSize, unsigned long *numOfElements);
	void getPointFromIndex(unsigned long index, DATA_TYPE* result);

	template<class ImplementedModel>
	void slaveProcess();

	template<class ImplementedModel>
	void computeThread(ComputeThreadInfo& cti);
};

template<class ImplementedModel>
int ParallelFramework::run() {
	if(!valid){
		printf("[%d] run() called for invalid framework\n", rank);
		return -1;
	}

	if(rank == 0){

		printf("[%d] Master process starting\n", rank);
		masterProcess();
		printf("[%d] Master process finished\n", rank);

	}else{

		printf("[%d] Slave process starting\n", rank);
		slaveProcess<ImplementedModel>();
		printf("[%d] Slave process finished\n", rank);

	}

	MPI_Finalize();

	return 0;
}

template<class ImplementedModel>
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
	int listIndex;
	sem_t semResults;
	sem_init(&semResults, 0, 0);

	ComputeThreadInfo* computeThreadInfo = new ComputeThreadInfo[numOfThreads];
	for(int i=0; i<numOfThreads; i++){
		sem_init(&computeThreadInfo[i].semData, 0, 0);
		computeThreadInfo[i].semResults = &semResults;
		computeThreadInfo[i].results = nullptr;
		computeThreadInfo[i].listIndexPtr = &listIndex;
		computeThreadInfo[i].ratio = (float)1/numOfThreads;
	}

	ImplementedModel model_p = ImplementedModel();

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
			coordinatorThread(computeThreadInfo, omp_get_num_threads()-1, &model_p);
		}else{
			// Calculate id: -1 -> CPU, 0+ -> GPU[id]
			computeThreadInfo[tid-1].id = tid - (parameters->processingType == PROCESSING_TYPE_GPU ? 1 : 2);

			computeThread<ImplementedModel>(computeThreadInfo[tid - 1]);
		}
	}

	/*******************************************************************
	***************************** Finalize *****************************
	********************************************************************/
	sem_destroy(&semResults);
	for(int i=0; i<numOfThreads; i++){
		sem_destroy(&computeThreadInfo[i].semData);
	}
	delete[] computeThreadInfo;
}

template<class ImplementedModel>
void ParallelFramework::computeThread(ComputeThreadInfo& cti){
	int gpuListIndex, globalListIndexOld;

	// GPU Memory
	ImplementedModel** deviceModelAddress;	// GPU Memory to save the address of the 'Model' object on device
	RESULT_TYPE* deviceResults;				// GPU Memory for results
	int* deviceListIndexPtr;				// GPU Memory for list index for synchronization when saving the results as a list of points
	unsigned long long* deviceIdxSteps;		// GPU Memory to store idxSteps
	Limit* deviceLimits;					// GPU Memory to store the Limit structures
	void* deviceDataPtr;					// GPU Memory to store any constant data

	// GPU Runtime
	cudaStream_t streams[parameters->gpuStreams];
	unsigned long allocatedElements = 0;
	cudaDeviceProp deviceProp;
	bool useSharedMemory;

	/*******************************************************************
	************************* Initialization ***************************
	********************************************************************/

	// Initialize device
	if (cti.id > -1) {
		// Select gpu[id]
		cudaSetDevice(cti.id);

		// Get device's properties for shared memory
		cudaGetDeviceProperties(&deviceProp, cti.id);
		useSharedMemory = parameters->dataSize <= deviceProp.sharedMemPerBlock;

		// Create streams
		for(int i=0; i<parameters->gpuStreams; i++){
			cudaStreamCreate(&streams[i]);
			cce();
		}

		// Allocate memory on device
		cudaMalloc(&deviceModelAddress, sizeof(ImplementedModel**));				cce();
		cudaMalloc(&deviceResults, allocatedElements * sizeof(RESULT_TYPE));		cce();
		cudaMalloc(&deviceListIndexPtr, sizeof(int));								cce();
		cudaMalloc(&deviceLimits, parameters->D * sizeof(Limit));					cce();
		cudaMalloc(&deviceIdxSteps, parameters->D * sizeof(unsigned long long));			cce();
		if(parameters->dataSize > 0){
			cudaMalloc(&deviceDataPtr, parameters->dataSize);							cce();
		}

		// Instantiate the model object on the device, and write its address in 'deviceModelAddress' on the device
		create_model_kernel<ImplementedModel><<< 1, 1 >>>(deviceModelAddress);		cce();

		// Copy limits, idxSteps, and constant data to device
		cudaMemcpy(deviceLimits, limits, parameters->D * sizeof(Limit), cudaMemcpyHostToDevice);					cce();
		cudaMemcpy(deviceIdxSteps, idxSteps, parameters->D * sizeof(unsigned long long), cudaMemcpyHostToDevice);	cce();
		if(parameters->dataSize > 0){
			cudaMemcpy(deviceDataPtr, parameters->dataPtr, parameters->dataSize, cudaMemcpyHostToDevice);			cce();
		}

		#ifdef DBG_MEMORY
			printf("[%d] ComputeThread %d: deviceModelAddress: 0x%x\n", rank, cti.id, (void*) deviceModelAddress);
			printf("[%d] ComputeThread %d: deviceResults: 0x%x\n", rank, cti.id, (void*) deviceResults);
			printf("[%d] ComputeThread %d: deviceListIndexPtr: 0x%x\n", rank, cti.id, (void*) deviceListIndexPtr);
			printf("[%d] ComputeThread %d: deviceLimits: 0x%x\n", rank, cti.id, (void*) deviceLimits);
			printf("[%d] ComputeThread %d: deviceDataPtr: 0x%x\n", rank, cti.id, (void*) deviceDataPtr);
		#endif
	}


	/*******************************************************************
	************************* Execution loop ***************************
	********************************************************************/
	while(true){
		/*****************************************************************
		************* Request data from coordinateThread *****************
		******************************************************************/
		sem_wait(&cti.semData);
		cti.stopwatch.start();

		// If more data available...
		if (cti.numOfElements > 0) {
			#ifdef DBG_START_STOP
				printf("[%d] ComputeThread %d: Running for %ld elements...\n", rank, cti.id, cti.numOfElements);
			#endif
			#ifdef DBG_DATA
				printf("[%d] ComputeThread %d: Got %ld elements starting from %ld\n", rank, cti.id, cti.numOfElements, cti.startPoint);
			#endif
			fflush(stdout);

			/*****************************************************************
			 If batchSize was increased, allocate more memory for the results
			******************************************************************/
			if (allocatedElements < cti.numOfElements && cti.id > -1 && allocatedElements < getDefaultGPUBatchSize()) {

				#ifdef DBG_MEMORY
					printf("[%d] ComputeThread %d: Allocating more GPU memory (%ld -> %ld elements, %ld MB)\n", rank,
							cti.id, allocatedElements, cti.numOfElements, (cti.numOfElements*sizeof(RESULT_TYPE)) / (1024 * 1024));
					fflush(stdout);
				#endif

				allocatedElements = min(cti.numOfElements, getDefaultGPUBatchSize());

				// Reallocate memory on device
				cudaFree(deviceResults);
				cce();
				cudaMalloc(&deviceResults, allocatedElements * sizeof(RESULT_TYPE));
				cce();

				#ifdef DBG_MEMORY
					printf("[%d] ComputeThread %d: deviceResults = 0x%x\n", rank, cti.id, deviceResults);
				#endif
			}

			/*****************************************************************
			******************** Calculate the results ***********************
			******************************************************************/
			// If GPU...
			if (cti.id > -1) {

				// Initialize the list index counter
				cudaMemset(deviceListIndexPtr, 0, sizeof(int));

				// Divide the chunk to smaller chunks to scatter accross streams
				unsigned long elementsPerStream = cti.numOfElements / parameters->gpuStreams;
				bool onlyOne = false;
				unsigned long skip = 0;
				if(elementsPerStream == 0){
					elementsPerStream = cti.numOfElements;
					onlyOne = true;
				}

				// Queue the chunks to the streams
				for(int i=0; i<parameters->gpuStreams; i++){
					// Adjust elementsPerStream for last stream (= total-queued)
					if(i == parameters->gpuStreams - 1){
						elementsPerStream = cti.numOfElements - skip;
					}else{
						elementsPerStream = min(elementsPerStream, cti.numOfElements - skip);
					}

					// Queue the kernel in stream[i] (each GPU thread gets COMPUTE_BATCH_SIZE elements to calculate)
					int gpuThreads = (elementsPerStream + parameters->computeBatchSize - 1) / parameters->computeBatchSize;
					int blockSize = min(parameters->blockSize, gpuThreads);
					int numOfBlocks = (gpuThreads + blockSize - 1) / blockSize;
					// Note: Point at the start of deviceResults, because the offset is calculated in the kernel
					validate_kernel<ImplementedModel><<<numOfBlocks, blockSize, useSharedMemory ? parameters->dataSize : 0, streams[i]>>>(
						deviceModelAddress, deviceResults, deviceLimits, cti.startPoint,
						parameters->D, deviceIdxSteps, elementsPerStream, skip, deviceDataPtr, parameters->dataSize, useSharedMemory,
						parameters->resultSaveType == SAVE_TYPE_ALL ? nullptr : deviceListIndexPtr, parameters->computeBatchSize
					);

					// Queue the memcpy in stream[i] only if we are saving as SAVE_TYPE_ALL (otherwise the results will be fetched at the end of the current computation)
					if(parameters->resultSaveType == SAVE_TYPE_ALL){
						cudaMemcpyAsync(&cti.results[skip], &deviceResults[skip], elementsPerStream*sizeof(RESULT_TYPE), cudaMemcpyDeviceToHost, streams[i]);
					}

					#ifdef DBG_QUEUE
						printf("[%d] ComputeThread %d: Queueing %ld elements in stream %d (%d gpuThreads, %d blocks, %d block size), with skip=%ld\n", rank,
								cti.id, elementsPerStream, i, gpuThreads, numOfBlocks, blockSize, skip);
					#endif

					// Increase skip
					skip += elementsPerStream;

					if(onlyOne)
						break;
				}

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
					globalListIndexOld = __sync_fetch_and_add(cti.listIndexPtr, gpuListIndex);

					// Get the results from the GPU
					cudaMemcpy(&((DATA_TYPE*)cti.results)[globalListIndexOld], deviceResults, gpuListIndex * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
				}


			} else {

				cpu_kernel<ImplementedModel>(cti.results, limits, parameters->D, cti.numOfElements, parameters->dataPtr, parameters->resultSaveType == SAVE_TYPE_ALL ? nullptr : cti.listIndexPtr,
				idxSteps, cti.startPoint);

			}

			#ifdef DBG_RESULTS
				if(parameters->resultSaveType == SAVE_TYPE_ALL){
					printf("[%d] ComputeThread %d: Results are: ", rank, cti.id);
					for (int i = 0; i < cti.numOfElements; i++) {
						printf("%f ", ((DATA_TYPE *)cti.results)[i]);
					}
					printf("\n");
				}
			#endif

		}
		// else if computation is complete...
		else if(cti.startPoint == 0){
			// No more data, exit
			break;
		}
		// else 0 elements were assigned due to extremely low computation score (improbable)
		else{
			// Sleep 1 ms to prevent stopwatch time = 0
			usleep(1000);
		}

		// Stop the stopwatch
		cti.stopwatch.stop();

		// Let coordinatorThread know that the results are ready
		sem_post(cti.semResults);
	}

	/*******************************************************************
	 *************************** Finalize ******************************
	 *******************************************************************/

	// Finalize GPU
	if (cti.id > -1) {
		// Delete the model object on the device
		delete_model_kernel<ImplementedModel><<<1, 1 >>>(deviceModelAddress);
		cce();

		// Free the space for the model's address on the device
		cudaFree(deviceModelAddress);		cce();
		cudaFree(deviceResults);			cce();
		cudaFree(deviceListIndexPtr);		cce();
		cudaFree(deviceLimits);				cce();
		cudaFree(deviceIdxSteps);			cce();
		cudaFree(deviceDataPtr);			cce();

		// Make sure streams are finished and destroy them
		for(int i=0;i<parameters->gpuStreams;i++){
			cudaStreamDestroy(streams[i]);
			cce();
		}
	}
}

#endif
