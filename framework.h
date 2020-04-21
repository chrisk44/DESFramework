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
	Limit* limits = NULL;			// This must be an array of length = parameters.D
	ParallelFrameworkParameters* parameters = NULL;

	// Runtime variables
	unsigned long* idxSteps = NULL;			// Index steps for each dimension
	RESULT_TYPE* finalResults = NULL;		// An array of N0 * N1 * ... * N(D-1)
	bool valid = false;
	unsigned long* toSendVector = NULL;		// An array of D elements, where every entry shows the next element of that dimension to be dispatched
	unsigned long totalSent = 0;			// Total elements that have been sent for processing
	unsigned long totalReceived = 0;		// TOtal elements that have been calculated and returned
	unsigned long totalElements = 0;		// Total elements

	// MPI
	int rank;

public:
	ParallelFramework(Limit* limits, ParallelFrameworkParameters& parameters);
	~ParallelFramework();

	template<class ImplementedModel>
	int run();

	RESULT_TYPE* getResults();
	void getIndicesFromPoint(DATA_TYPE* point, unsigned long* dst);
	long getIndexFromIndices(unsigned long* pointIdx);
	bool isValid();

private:
	void masterProcess();
	void coordinatorThread(ComputeThreadInfo* cti, int numOfThreads);
	void getDataChunk(unsigned long maxBatchSize, unsigned long* toCalculate, int *numOfElements);
	void addToIdxVector(unsigned long* start, unsigned long* result, int num, int* overflow);

	template<class ImplementedModel>
	void slaveProcess();

	template<class ImplementedModel>
	void computeThread(ComputeThreadInfo& cti);
};

template<class ImplementedModel>
int ParallelFramework::run() {
	// Initialize MPI
	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(rank == 0){

		printf("Master %d process starting\n", rank);
		masterProcess();
		printf("Master %d process finished\n", rank);

	}else{

		printf("Slave %d process starting\n", rank);
		slaveProcess<ImplementedModel>();
		printf("Slave %d process finished\n", rank);

	}

	MPI_Finalize();

	// If we are a slave, stop here
	if(rank != 0){
		printf("run() exiting\n");
		exit(0);
	}

	return 0;
}

template<class ImplementedModel>
void ParallelFramework::slaveProcess() {
	/*******************************************************************
	 ********* Calculate number of worker threads (#GPUs + 1CPU) *******
	********************************************************************/
	int numOfThreads = 0;

	if(parameters->processingType != TYPE_CPU)
		cudaGetDeviceCount(&numOfThreads);

	if(parameters->processingType != TYPE_GPU)
		numOfThreads++;

	if(numOfThreads == 0){
		printf("[E] SlaveProcess %d: numOfThreads is 0\n", rank);
		return;
	}


	/*******************************************************************
	 ********************* Initialization ******************************
	********************************************************************/
	sem_t semResults;
	sem_init(&semResults, 0, 0);

	ComputeThreadInfo* computeThreadInfo = new ComputeThreadInfo[numOfThreads];
	for(int i=0; i<numOfThreads; i++){
		sem_init(&computeThreadInfo[i].semData, 0, 0);
		computeThreadInfo[i].semResults = &semResults;
		computeThreadInfo[i].results = nullptr;
		computeThreadInfo[i].startPointIdx = new unsigned long[parameters->D];
		computeThreadInfo[i].ratio = (float)1/numOfThreads;
		computeThreadInfo[i].stopwatch.reset();
	}

	#if DEBUG >= 1
		printf("SlaveProcess %d: Spawning %d worker threads...\n", rank, numOfThreads);
	#endif

	/*******************************************************************
	 ************** Launch coordinator and worker threads **************
	********************************************************************/
	#pragma omp parallel num_threads(numOfThreads + 1) shared(computeThreadInfo) 	// +1 thread to handle the communication with masterProcess
	{
		int tid = omp_get_thread_num();
		if(tid == 0){
			coordinatorThread(computeThreadInfo, omp_get_num_threads()-1);
		}else{
			// Calculate id: -1 -> CPU, 0+ -> GPU[id]
			computeThreadInfo[tid-1].id = tid - (parameters->processingType == TYPE_GPU ? 1 : 2);

			computeThread<ImplementedModel>(computeThreadInfo[tid - 1]);
		}
	}

	/*******************************************************************
	 **************************** Finalize *****************************
	********************************************************************/
	sem_destroy(&semResults);
	for(int i=0; i<numOfThreads; i++){
		sem_destroy(&computeThreadInfo[i].semData);
		delete[] computeThreadInfo[i].startPointIdx;
	}
	delete[] computeThreadInfo;
}

template<class ImplementedModel>
void ParallelFramework::computeThread(ComputeThreadInfo& cti){

	// GPU Memory
	ImplementedModel** deviceModelAddress;	// GPU Memory to save the address of the 'Model' object on device
	RESULT_TYPE* deviceResults;				// GPU Memory for results
	unsigned long* deviceStartingPointIdx;	// GPU Memory to store the start point indices
	Limit* deviceLimits;					// GPU Memory to store the Limit structures

	// GPU Runtime
	cudaStream_t streams[NUM_OF_STREAMS];
	int allocatedElements = 500;

	/*******************************************************************
	 ************************ Initialization ***************************
	********************************************************************/

	// Initialize device
	if (cti.id > -1) {
		// Select gpu[id]
		cudaSetDevice(cti.id);

		// Create streams
		for(int i=0; i<NUM_OF_STREAMS; i++){
			cudaStreamCreate(&streams[i]);
			cce();
		}

		// Allocate memory on device
		cudaMalloc(&deviceModelAddress, sizeof(ImplementedModel**));				cce();
		cudaMalloc(&deviceResults, allocatedElements * sizeof(RESULT_TYPE));		cce();
		cudaMalloc(&deviceStartingPointIdx, parameters->D * sizeof(unsigned long));	cce();
		cudaMalloc(&deviceLimits, parameters->D * sizeof(Limit));					cce();

		// Instantiate the model object on the device, and write its address in 'deviceModelAddress' on the device
		create_model_kernel<ImplementedModel><<< 1, 1 >>>(deviceModelAddress);	cce();

		// Copy limits to device
		cudaMemcpy(deviceLimits, limits, parameters->D * sizeof(Limit), cudaMemcpyHostToDevice);	cce();

		#if DEBUG > 2
			printf("ComputeThread %d %d: deviceModelAddress: 0x%x\n", rank, cti.id, (void*) deviceModelAddress);
			printf("ComputeThread %d %d: deviceResults: 0x%x\n", rank, cti.id, (void*) deviceResults);
			printf("ComputeThread %d %d: deviceStartingPointIdx: 0x%x\n", rank, cti.id, (void*) deviceStartingPointIdx);
			printf("ComputeThread %d %d: deviceLimits: 0x%x\n", rank, cti.id, (void*) deviceLimits);
		#endif
	}



	/*******************************************************************
	 ************************ Execution loop ***************************
	********************************************************************/

	while(true){
		//
		// Wait for data from coordinateThread
		//
		sem_wait(&cti.semData);
		cti.stopwatch.start();

		// If more data available...
		if (cti.numOfElements > 0) {
			#if DEBUG >= 1
				printf("ComputeThread %d %d: Running for %d elements...\n", rank, cti.id, cti.numOfElements);
			#elif DEBUG >= 3
				printf("ComputeThread %d %d: Got %d elements starting from  ", rank, cti.id, cti.numOfElements);
				for (unsigned int i = 0; i < parameters->D; i++)
					printf("%d ", cti.startPointIdx[i]);
				printf("\n");
			#endif
			fflush(stdout);

			//
			// If batchSize was increased, allocate more memory for the results
			//
			if (allocatedElements < cti.numOfElements && cti.id > -1) {
				#if DEBUG >= 2
					printf("ComputeThread %d %d: Allocating more GPU memory (%d -> %d elements, %ld MB)\n", rank,
							cti.id, allocatedElements, cti.numOfElements, (cti.numOfElements*sizeof(RESULT_TYPE)) / (1024 * 1024));
					fflush(stdout);
				#endif

				allocatedElements = cti.numOfElements;

				// Reallocate memory on device
				cudaFree(deviceResults);
				cce();
				cudaMalloc(&deviceResults, allocatedElements * sizeof(RESULT_TYPE));
				cce();

				#if DEBUG >=2
					printf("ComputeThread %d %d: deviceResults = 0x%x\n", rank, cti.id, deviceResults);
				#endif
			}

			//
			// Calculate the results
			//
			// If GPU...
			if (cti.id > -1) {

				// Copy starting point indices to device
				cudaMemcpy(deviceStartingPointIdx, cti.startPointIdx, parameters->D * sizeof(unsigned long), cudaMemcpyHostToDevice);
				cce();

				// Divide the chunk to smaller chunks to scatter accross streams
				int elementsPerStream = cti.numOfElements / NUM_OF_STREAMS;
				bool onlyOne = false;
				int skip = 0;
				if(elementsPerStream == 0){
					elementsPerStream = cti.numOfElements;
					onlyOne = true;
				}

				// Queue the chunks to the streams
				for(int i=0; i<NUM_OF_STREAMS; i++){
					// Adjust elementsPerStream for last stream (= total-queued)
					if(i == NUM_OF_STREAMS - 1){
						elementsPerStream = cti.numOfElements - skip;
					}else{
						elementsPerStream = min(elementsPerStream, cti.numOfElements - skip);
					}

					// Queue the kernel in stream[i] (each GPU thread gets COMPUTE_BATCH_SIZE elements to calculate)
					int gpuThreads = (elementsPerStream + COMPUTE_BATCH_SIZE - 1) / COMPUTE_BATCH_SIZE;
					int blockSize = min(BLOCK_SIZE, gpuThreads);
					int numOfBlocks = (gpuThreads + blockSize - 1) / blockSize;
					validate_kernel<ImplementedModel><<<numOfBlocks, blockSize, 0, streams[i]>>>(		// Note: Point at the start of deviceResults, because the offset is calculated in the kernel
						deviceModelAddress, deviceStartingPointIdx, deviceResults, deviceLimits, parameters->D, elementsPerStream, skip
					);

					// Queue the memcpy in stream[i]
					cudaMemcpyAsync(&cti.results[skip], &deviceResults[skip], elementsPerStream*sizeof(RESULT_TYPE), cudaMemcpyDeviceToHost, streams[i]);

					#if DEBUG >= 2
					printf("ComputeThread %d %d: Queueing %d elements in stream %d (%d gpuThreads, %d blocks, %d block size), with skip=%d\n", rank,
								cti.id, elementsPerStream, i, gpuThreads, numOfBlocks, blockSize, skip);
					#endif

					// Increase skip
					skip += elementsPerStream;

					if(onlyOne)
						break;
				}

				// Wait for all streams to finish
				for(int i=0; i<NUM_OF_STREAMS; i++){
					cudaStreamSynchronize(streams[i]);
					cce();
				}


			} else {

				cpu_kernel<ImplementedModel>(cti.startPointIdx, cti.results, limits, parameters->D, cti.numOfElements);

			}

			#if DEBUG >= 2
				printf("ComputeThread %d %d: Finished work\n", rank, cti.id);
			#endif
			#if DEBUG >= 4
				printf("ComputeThread %d %d: Results are: ", rank, cti.id);
				for (int i = 0; i < cti.numOfElements; i++) {
					printf("%f ", cti.results[i]);
				}
				printf("\n");
			#endif

		}
		// else if computation is complete...
		else if(cti.numOfElements == -1){
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
	********************************************************************/

	// Finalize GPU
	if (cti.id > -1) {
		// Delete the model object on the device
		delete_model_kernel<ImplementedModel><<<1, 1 >>>(deviceModelAddress);
		cce();

		// Free the space for the model's address on the device
		cudaFree(deviceModelAddress);		cce();
		cudaFree(deviceResults);			cce();
		cudaFree(deviceStartingPointIdx);	cce();
		cudaFree(deviceLimits);				cce();

		// Make sure streams are finished and destroy them
		for(int i=0;i<NUM_OF_STREAMS;i++){
			cudaStreamDestroy(streams[i]);
			cce();
		}
	}
}

#endif
