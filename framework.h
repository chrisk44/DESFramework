#ifndef PARALLELFRAMEWORK_H
#define PARALLELFRAMEWORK_H

#include <cuda.h>
#include <iostream>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
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
	RESULT_TYPE* results = NULL;			// An array of N0 * N1 * ... * N(D-1)
	bool valid = false;
	unsigned long* toSendVector = NULL;		// An array of D elements, where every entry shows the next element of that dimension to be dispatched
	unsigned long totalSent = 0;			// Total elements that have been sent for processing
	unsigned long totalReceived = 0;		// TOtal elements that have been calculated and returned
	unsigned long totalElements = 0;		// Total elements

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
	void coordinatorThread(ProcessingThreadInfo* pti, int numOfThreads);
	void getDataChunk(unsigned long maxBatchSize, unsigned long* toCalculate, int *numOfElements);
	void addToIdxVector(unsigned long* start, unsigned long* result, int num, int* overflow);

	template<class ImplementedModel>
	void slaveProcess();

	template<class ImplementedModel>
	void computeThread(ProcessingThreadInfo& pti);
};

template<class ImplementedModel>
int ParallelFramework::run() {
	// Initialize MPI
	int rank;
	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(rank == 0){

		printf("Master process starting\n");
		masterProcess();
		printf("Master process finished\n");

	}else{

		printf("Slave process starting\n");
		slaveProcess<ImplementedModel>();
		printf("Slave process finished\n");

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
	// Calculate number of threads (#GPUs + 1CPU)
	int numOfThreads = 0;

	if(parameters->processingType != TYPE_CPU)
		cudaGetDeviceCount(&numOfThreads);

	if(parameters->processingType != TYPE_GPU)
		numOfThreads++;

	if(numOfThreads == 0){
		printf("[E] SlaveProcess: numOfThreads is 0\n");
		return;
	}

	sem_t semResults;
	sem_init(&semResults, 0, 0);

	ProcessingThreadInfo* PTIs = new ProcessingThreadInfo[numOfThreads];

	// Initialize PTIs
	for(int i=0; i<numOfThreads; i++){

		sem_init(&PTIs[i].semData, 0, 0);
		PTIs[i].semResults = &semResults;
		PTIs[i].results = nullptr;
		PTIs[i].startPointIdx = new unsigned long[parameters->D];
	}

	#if DEBUG >= 1
		printf("SlaveProcess: Spawning %d worker threads...\n", numOfThreads);
	#endif

	#pragma omp parallel num_threads(numOfThreads + 1) shared(PTIs) 	// +1 thread to handle the communication with masterProcess
	{
		int tid = omp_get_thread_num();
		if(tid == 0){
			coordinatorThread(PTIs, omp_get_num_threads()-1);
		}else{
			// Calculate id: -1 -> CPU, 0+ -> GPU[id]
			PTIs[tid-1].id = tid - (parameters->processingType == TYPE_GPU ? 1 : 2);

			computeThread<ImplementedModel>(PTIs[tid - 1]);
		}
	}

	// Free resources
	for(int i=0; i<numOfThreads; i++){
		sem_destroy(&PTIs[i].semData);
		delete[] PTIs[i].startPointIdx;
	}

	sem_destroy(&semResults);
}

template<class ImplementedModel>
void ParallelFramework::computeThread(ProcessingThreadInfo& pti){

	// GPU Memory
	ImplementedModel** deviceModelAddress;	// GPU Memory to save the address of the 'Model' object on device
	RESULT_TYPE* deviceResults;				// GPU Memory for results
	unsigned long* deviceStartingPointIdx;	// GPU Memory to store the start point indices
	Limit* deviceLimits;					// GPU Memory to store the Limit structures

	// GPU Runtime
	cudaStream_t streams[NUM_OF_STREAMS];
	int blockSize;							// Size of thread blocks
	int numOfBlocks;						// Number of blocks
	size_t freeMem, totalMem;				// Bytes of free,total memory on GPU
	int allocatedElements = 500;

	// Initialize device
	if (pti.id > -1) {
		// Select gpu[id]
		cudaSetDevice(pti.id);

		// Create streams
		for(int i=0; i<NUM_OF_STREAMS; i++){
			cudaStreamCreate(&streams[i]);
			cce();
		}

		// Read device's memory info
		cudaMemGetInfo(&freeMem, &totalMem);
		//maxBatchSize = (freeMem - MEM_GPU_SPARE_BYTES) / sizeof(RESULT_TYPE);

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
			printf("ComputeThread %d: deviceModelAddress: 0x%x\n", pti.id, (void*) deviceModelAddress);
			printf("ComputeThread %d: deviceResults: 0x%x\n", pti.id, (void*) deviceResults);
			printf("ComputeThread %d: deviceStartingPointIdx: 0x%x\n", pti.id, (void*) deviceStartingPointIdx);
			printf("ComputeThread %d: deviceLimits: 0x%x\n", pti.id, (void*) deviceLimits);
		#endif
	}

	// #if DEBUG >= 2
	// 	printf("ComputeThread %d: maxBatchSize = %d (%ld MB)\n", id, maxBatchSize, maxBatchSize*sizeof(RESULT_TYPE) / (1024 * 1024));
	// #endif

	while(true){
		// Wait for data from coordinateThread
		sem_wait(&pti.semData);

		// If more data available...
		if (pti.numOfElements > 0) {
			#if DEBUG >= 1
				printf("ComputeThread %d: Running for %d elements...\n", pti.id, pti.numOfElements);
			#endif
			#if DEBUG >= 3
				printf("ComputeThread %d: Got %d elements starting from  ", pti.id, pti.numOfElements);
				for (unsigned int i = 0; i < parameters->D; i++)
					printf("%d ", pti.startPointIdx[i]);
				printf("\n");
			#endif
			fflush(stdout);

			// If batchSize was increased, allocate more memory for the results
			if (allocatedElements < pti.numOfElements && pti.id > -1) {
				#if DEBUG >= 2
					printf("ComputeThread %d: Allocating more GPU memory (%d -> %d elements, %ld MB)\n",
							pti.id, allocatedElements, pti.numOfElements, (pti.numOfElements*sizeof(RESULT_TYPE)) / (1024 * 1024));
					fflush(stdout);
				#endif

				allocatedElements = pti.numOfElements;

				// Reallocate memory on device
				cudaFree(deviceResults);
				cce();
				cudaMalloc(&deviceResults, allocatedElements * sizeof(RESULT_TYPE));
				cce();

				#if DEBUG >=2
					printf("ComputeThread %d: deviceResults = 0x%x\n", pti.id, deviceResults);
				#endif
			}

			// Calculate the results
			if (pti.id > -1) {

				// Copy starting point indices to device
				cudaMemcpy(deviceStartingPointIdx, pti.startPointIdx, parameters->D * sizeof(unsigned long), cudaMemcpyHostToDevice);
				cce();

				int elementsPerStream = pti.numOfElements / NUM_OF_STREAMS;
				bool onlyOne = false;
				int skip = 0;
				if(elementsPerStream == 0){
					elementsPerStream = pti.numOfElements;
					onlyOne = true;
				}
				for(int i=0; i<NUM_OF_STREAMS; i++){
					// Adjust elementsPerStream for last stream (= total-queued)
					if(i == NUM_OF_STREAMS - 1){
						elementsPerStream = pti.numOfElements - skip;
					}else{
						elementsPerStream = min(elementsPerStream, pti.numOfElements - skip);
					}

					// Queue the kernel in stream[i]
					int gpuThreads = (elementsPerStream + COMPUTE_BATCH_SIZE - 1) / COMPUTE_BATCH_SIZE;
					blockSize = min(BLOCK_SIZE, gpuThreads);
					numOfBlocks = (gpuThreads + blockSize - 1) / blockSize;
					validate_kernel<ImplementedModel><<<numOfBlocks, blockSize, 0, streams[i]>>>(		// Note: Point at the start of deviceResults, because the offset is calculated in the kernel
						deviceModelAddress, deviceStartingPointIdx, deviceResults, deviceLimits, parameters->D, elementsPerStream, skip
					);

					// Queue the memcpy in stream[i]
					cudaMemcpyAsync(&pti.results[skip], &deviceResults[skip], elementsPerStream*sizeof(RESULT_TYPE), cudaMemcpyDeviceToHost, streams[i]);

					#if DEBUG >= 2
					printf("ComputeThread %d: Queueing %d elements in stream %d (%d gpuThreads, %d blocks, %d block size), with skip=%d\n",
								pti.id, elementsPerStream, i, gpuThreads, numOfBlocks, blockSize, skip);
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

				cpu_kernel<ImplementedModel>(pti.startPointIdx, pti.results, limits, parameters->D, pti.numOfElements);

			}

			#if DEBUG >= 4
				printf("ComputeThread %d: Results are: ", pti.id);
				for (int i = 0; i < pti.numOfElements; i++) {
					printf("%f ", pti.results[i]);
				}
				printf("\n");
			#endif

			// Let coordinatorThread know that the results are ready
			sem_post(pti.semResults);

		} else {
			// No more data
			break;
		}
	}

	// Finalize GPU
	if (pti.id > -1) {
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
