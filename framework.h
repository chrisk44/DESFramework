#ifndef PARALLELFRAMEWORK_H
#define PARALLELFRAMEWORK_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <Windows.h>

#include "utilities.h"
#include "kernels.cpp"

using namespace std;

class ParallelFramework {
public:
	// Parameters
	Limit* limits = NULL;			// This must be an array of length = parameters.D
	ParallelFrameworkParameters* parameters = NULL;

	// Runtime variables
	unsigned long* idxSteps = NULL;			// Index steps for each dimension
	DATA_TYPE* steps = NULL;				// Real step for each dimension
	RESULT_TYPE* results = NULL;			// An array of N0 * N1 * ... * N(D-1)
	bool valid = false;
	unsigned long* toSendVector = NULL;		// An array of D elements, where every entry shows the next element of that dimension to be dispatched
	unsigned long totalSent = 0;			// Total elements that have been sent for processing
	unsigned long totalElements = 0;		// Total elements

public:
	ParallelFramework(Limit* limits, ParallelFrameworkParameters& parameters);
	~ParallelFramework();

	template<class ImplementedModel>
	int run(char* argv0);

	RESULT_TYPE* getResults();
	void getIndicesFromPoint(DATA_TYPE* point, unsigned long* dst);
	long getIndexFromIndices(unsigned long* pointIdx);
	bool isValid();

public:
	int masterThread(MPI_Comm& comm, int numOfProcesses);

	template<class ImplementedModel>
	int slaveThread(int rank, MPI_Comm& comm);

	void getDataChunk(unsigned long maxBatchSize, unsigned long* toCalculate, int *numOfElements);

};

template<class ImplementedModel>
int ParallelFramework::run(char* argv0) {
	int numOfProcesses, tmp;
	char* programName;
	// MPI variables
	int rank;
	int* errcodes;
	MPI_Comm parentcomm, intercomm;

	// Initialize MPI
	MPI_Init(nullptr, nullptr);
	MPI_Comm_get_parent(&parentcomm);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Calculate number of processes (#GPUs + 1CPU)
	numOfProcesses = 0;

	if(parameters->processingType != TYPE_CPU)
		cudaGetDeviceCount(&numOfProcesses);

	if(parameters->processingType != TYPE_GPU)
		numOfProcesses++;

	// Allocate errcodes now that numOfProcesses is known
	errcodes = new int[numOfProcesses];

	// Isolate the program's name from argv0
	tmp = (int)strlen(argv0) - 1;
	while (argv0[tmp] != '/' && argv0[tmp] != '\\') {
		tmp--;
	}
	programName = &argv0[tmp+1];

	// If this is the parent process, spawn children and run masterThread, else run slaveThread
	if (parentcomm == MPI_COMM_NULL) {
#if DEBUG >=1
		cout << "Master: Spawning " << numOfProcesses << " processes" << endl;
#endif

		MPI_Comm_spawn(programName, MPI_ARGV_NULL, numOfProcesses, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &intercomm, errcodes);

		// Check errorcodes
		for (int i = 0; i < numOfProcesses; i++) {
			if (errcodes[i] != MPI_SUCCESS) {
				cout << "[E] Error starting process " << i << ", error: " << errcodes[i] << endl;

				// TBD: Terminate everyone (?)
				//totalSent = totalElements;
			}
		}

		masterThread(intercomm, numOfProcesses);
#if DEBUG >=1
		cout << "Master finished" << endl;
#endif

	} else {

#if DEBUG >=1
		cout << "Slave " << rank << " starting" << endl;
#endif
		slaveThread<ImplementedModel>(rank, parentcomm);
#if DEBUG >=1
		cout << "Slave " << rank << " finished" << endl;
#endif

		MPI_Finalize();
		exit(0);

	}

	// Only parent (master) continues here
	MPI_Finalize();

	delete[] errcodes;

	return 0;
}

template<class ImplementedModel>
int ParallelFramework::slaveThread(int rank, MPI_Comm& comm) {
	// Rank 0 is cpu, 1+ are gpus

	int gpuId = parameters->processingType == TYPE_GPU ? rank : rank - 1;

	unsigned long* startPointIdx = new unsigned long[parameters->D];
	RESULT_TYPE* tmpResults = new RESULT_TYPE[parameters->batchSize];			// TODO: batchSize might become larger, need to allocate more memory
	unsigned long numOfElements;
	int blockSize;
	int numOfBlocks;
	MPI_Status status;

	// deviceModelAddress is where the device address of the model is saved
	ImplementedModel** deviceModelAddress;

	RESULT_TYPE* deviceResults;
	unsigned long* deviceStartingPointIdx;
	Limit* deviceLimits;

	// Initialize GPU (instantiate an ImplementedModel object on the device)
	if (gpuId > -1) {
		size_t freeMem, totalMem;
		size_t toAllocate = (sizeof(ImplementedModel**) + parameters->batchSize * sizeof(RESULT_TYPE) + parameters->D * (sizeof(unsigned long) + sizeof(Limit)));
		// Select GPU with 'id'
		cudaSetDevice(gpuId);

		// Read memory info
		cudaMemGetInfo(&freeMem, &totalMem);

#if DEBUG >= 1
		printf("Allocating %d bytes on GPU %d (GPU Free Memory: %d/%d MB)\n", toAllocate, gpuId, (freeMem/1024)/1024, (totalMem/1024)/1024);
#endif

		// Allocate memory on device
		cudaMalloc(&deviceModelAddress, sizeof(ImplementedModel**));				cce();
		cudaMalloc(&deviceResults, parameters->batchSize * sizeof(RESULT_TYPE));	cce();	// TODO: batchSize might become larger, need to allocate more memory
		cudaMalloc(&deviceStartingPointIdx, parameters->D * sizeof(unsigned long));	cce();
		cudaMalloc(&deviceLimits, parameters->D * sizeof(Limit));					cce();

		// Create the model object on the device, and write its address in 'deviceModelAddress' on the device
		create_model_kernel<ImplementedModel> << < 1, 1 >> > (deviceModelAddress);	cce();

		// Move limits to device
		cudaMemcpy(deviceLimits, limits, parameters->D * sizeof(Limit), cudaMemcpyHostToDevice);	cce();

#if DEBUG > 2
		printf("\ndeviceModelAddress: 0x%x\n", (void*) deviceModelAddress);
		printf("deviceResults: 0x%x\n", (void*) deviceResults);
		printf("deviceStartingPointIdx: 0x%x\n", (void*) deviceStartingPointIdx);
		printf("deviceLimits: 0x%x\n", (void*) deviceLimits);
#endif
	}

	while (true) {

		// Send 'ready' signal to master
#if DEBUG >= 2
		cout << "  Slave " << rank << " sending READY..." << endl;
#endif
		MPI_Send(nullptr, 0, MPI_INT, 0, TAG_READY, comm);

		// Receive data (length and starting point) to compute
#if DEBUG >= 2
		cout << "  Slave " << rank << " waiting for data..." << endl;
#endif
		MPI_Recv(&numOfElements, 1, MPI_UNSIGNED_LONG, 0, TAG_DATA_COUNT, comm, &status);
		MPI_Recv(startPointIdx, parameters->D, MPI_UNSIGNED_LONG, 0, TAG_DATA, comm, &status);

		// If received more data...
		if (numOfElements > 0) {
#if DEBUG >= 3
			cout << "Got " << numOfElements << " elements: ";
			for (unsigned int i = 0; i < parameters->D; i++)
				cout << startPointIdx[i] << " ";
			cout << endl;
#endif
				
			// Calculate the results
			if (gpuId > -1) {

				//if (startPointIdx[0] != 0) {
					// Copy starting point indices to device
					cudaMemcpy(deviceStartingPointIdx, startPointIdx, parameters->D * sizeof(unsigned long), cudaMemcpyHostToDevice);
					cce();
					cudaDeviceSynchronize();

					// Call the kernel
					blockSize = min(BLOCK_SIZE, numOfElements);
					numOfBlocks = (numOfElements + blockSize - 1) / blockSize;
					printf("Starting %d blocks with blockSize %d\n", numOfBlocks, blockSize);
					validate_kernel<ImplementedModel> << <numOfBlocks, blockSize >> > (deviceModelAddress, deviceStartingPointIdx, deviceResults, deviceLimits, parameters->D, numOfElements);
					cce();

					// Wait for kernel to finish
					cudaDeviceSynchronize();
					cce();

					// Get results from device
					cudaMemcpy(tmpResults, deviceResults, numOfElements * sizeof(RESULT_TYPE), cudaMemcpyDeviceToHost);
					cce();
				//}
			} else {

				cpu_kernel<ImplementedModel>(startPointIdx, tmpResults, limits, parameters->D, numOfElements);

			}

			// Send the results to master
#if DEBUG >= 2
			cout << "  Slave " << rank << " sending " << numOfElements << " RESULTS..." << endl;
#endif
#if DEBUG >= 4
			// Print results
			printf("  Slave %d results: ", rank);
			for (int i = 0; i < numOfElements; i++) {
				printf("%f ", tmpResults[i]);
			}
			printf("\n");
#endif
			MPI_Send(tmpResults, numOfElements, RESULT_MPI_TYPE, 0, TAG_RESULTS, comm);

		} else {
			// No more data
#if DEBUG >= 2
			cout << "  Slave " << rank << " got 0 data..." << endl;
#endif
			break;
		}
	}

	// Finalize GPU
	if (gpuId > -1) {
		// Delete the model object on the device
		delete_model_kernel<ImplementedModel> << < 1, 1 >> > (deviceModelAddress);
		cce();

		// Free the space for the model's address on the device
		cudaFree(deviceModelAddress);		cce();
		cudaFree(deviceResults);			cce();
		cudaFree(deviceStartingPointIdx);	cce();
		cudaFree(deviceLimits);				cce();
	}

	delete[] startPointIdx;
	delete[] tmpResults;

	return 0;
}

#endif
