#ifndef PARALLELFRAMEWORK_H
#define PARALLELFRAMEWORK_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "utilities.h"
#include "kernels.cu"

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
	unsigned long totalReceived = 0;		// TOtal elements that have been calculated and returned
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
	void masterThread(MPI_Comm& comm, int* numOfProcesses);
	void listenerThread(MPI_Comm* finalcomm, int* numOfProcesses, bool* stopFlag);

	template<class ImplementedModel>
	void slaveThread(MPI_Comm& comm, int rank);

	void getDataChunk(unsigned long maxBatchSize, unsigned long* toCalculate, int *numOfElements);

};

template<class ImplementedModel>
int ParallelFramework::run(char* argv0) {
	// MPI variables
	int rank;
	MPI_Comm parentcomm, finalcomm;

	// Initialize MPI
	MPI_Init(nullptr, nullptr);
	MPI_Comm_get_parent(&parentcomm);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// If this is the parent process, spawn children and run masterThread, else run slaveThread
	if (parentcomm == MPI_COMM_NULL) {
		int* errcodes;
		int numOfProcesses;
		MPI_Comm intercomm;

		// Calculate number of processes (#GPUs + 1CPU)
		numOfProcesses = 0;

		if(parameters->processingType != TYPE_CPU)
			cudaGetDeviceCount(&numOfProcesses);

		if(parameters->processingType != TYPE_GPU)
			numOfProcesses++;

		#if DEBUG >=1
			cout << "Master: Spawning " << numOfProcesses << " processes" << endl;
		#endif

		// Allocate errcodes now that numOfProcesses is known
		errcodes = new int[numOfProcesses];

		char* argv[] = {
			"remote",
			nullptr
		};

		MPI_Comm_spawn(argv0, parameters->remote ? argv : MPI_ARGV_NULL, numOfProcesses, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &intercomm, errcodes);
		MPI_Intercomm_merge(intercomm, 0, &finalcomm);

		// Check errorcodes
		for (int i = 0; i < numOfProcesses; i++)
			if (errcodes[i] != MPI_SUCCESS)
				cout << "[E] Error starting process " << i << ", error: " << errcodes[i] << endl;


		delete[] errcodes;

		if (! (parameters->remote)) {
			bool stopFlag = false;
			#pragma omp parallel num_threads(2) shared(stopFlag)
			{
				if (omp_get_thread_num() == 0) {
					masterThread(finalcomm, &numOfProcesses);
					stopFlag = true;
				} else {
					listenerThread(&finalcomm, &numOfProcesses, &stopFlag);
				}
			}
		}

		MPI_Comm_free(&finalcomm);

		#if DEBUG >=1
		cout << "Master finished" << endl;
		#endif

	} else {

		#if DEBUG >=1
		cout << "Slave " << rank << " starting" << endl;
		#endif

		int sock = -1;
		if (parameters->remote) {
			// Connect to host at parameters->serverName : DEFAULT_PORT
			struct sockaddr_in serv_addr;

			// Create the socket
			sock = socket(AF_INET, SOCK_STREAM, 0);
			if( socket < 0 ){
		        printf("[E] Slave %d: Can't create socket\n", rank);
				goto finished;
		    }

			// Set serv_addr attributes
			// Convert IPv4 and IPv6 addresses from text to binary form
		    if( inet_pton(AF_INET, parameters->serverName.c_str(), &serv_addr.sin_addr) <= 0){
				printf("[E] Slave %d: Invalid server address or address not supported\n", rank);
				goto finished;
		    }
		    serv_addr.sin_family = AF_INET;
		    serv_addr.sin_port = htons(DEFAULT_PORT);

			if( connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0 ){
				printf("[E] Slave %d: Connection failed\n", rank);
				goto finished;
		    }

			// Join the host
			MPI_Comm joinedComm;
			MPI_Comm_join(sock, &joinedComm);
			MPI_Intercomm_merge(joinedComm, 1, &finalcomm);
		}else{
			MPI_Intercomm_merge(parentcomm, 1, &finalcomm);
		}

		slaveThread<ImplementedModel>(finalcomm, rank);

		// Free the instracommunicator
		MPI_Comm_free(&finalcomm);

		finished:

		#if DEBUG >=1
		cout << "Slave " << rank << " finished" << endl;
		#endif

		MPI_Finalize();

		if (sock != -1) {
			close(sock);
		}

		exit(0);

	}

	// Only parent (master) continues here
	MPI_Finalize();

	// If we are running in remote mode, master should stop here
	if(parameters->remote){
		printf("Master exiting\n");
		exit(0);
	}

	return 0;
}

template<class ImplementedModel>
void ParallelFramework::slaveThread(MPI_Comm& comm, int rank) {
	// If gpuId==-1, use CPU, otherwise use GPU with id = gpuId
	// Local ranks are >= 0
	int gpuId = rank - (parameters->processingType == TYPE_GPU ? 0 : 1);

	unsigned long allocatedElements = parameters->batchSize;										// Number of allocated elements for results
	RESULT_TYPE* tmpResults = (RESULT_TYPE*)malloc(allocatedElements * sizeof(RESULT_TYPE));		// Memory to save the results
	unsigned long* startPointIdx = new unsigned long[parameters->D];								// Memory to store the start point indices

	unsigned long maxBatchSize;				// Max batch size
	int numOfElements;						// Number of elements to process
	MPI_Status status;						// Structure to save the status of MPI communications

	// GPU parameters
	ImplementedModel** deviceModelAddress;	// GPU Memory to save the address of the 'Model' object on device
	RESULT_TYPE* deviceResults;				// GPU Memory for results
	unsigned long* deviceStartingPointIdx;	// GPU Memory to store the start point indices
	Limit* deviceLimits;					// GPU Memory to store the Limit structures
	int blockSize;							// Size of thread blocks
	int numOfBlocks;						// Number of blocks
	size_t freeMem, totalMem, toAllocate;	// Bytes of free,total,toAllocate memory on GPU

	// Initialize device
	if (gpuId > -1) {
		// Select device with id = gpuId
		cudaSetDevice(gpuId);

		// Read device's memory info
		cudaMemGetInfo(&freeMem, &totalMem);
		maxBatchSize = (freeMem - MEM_GPU_SPARE_BYTES) / sizeof(RESULT_TYPE);
		toAllocate = (sizeof(ImplementedModel**) + parameters->D * (sizeof(unsigned long) + sizeof(Limit))) + allocatedElements * sizeof(RESULT_TYPE);

#if DEBUG >= 1
		printf("  Slave %d: Allocating %d bytes on GPU %d (GPU Free Memory: %d/%d MB)\n", rank, toAllocate, gpuId, (freeMem/1024)/1024, (totalMem/1024)/1024);
#endif

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
		printf("  Slave %d: deviceModelAddress: 0x%x\n", rank, (void*) deviceModelAddress);
		printf("  Slave %d: deviceResults: 0x%x\n", rank, (void*) deviceResults);
		printf("  Slave %d: deviceStartingPointIdx: 0x%x\n", rank, (void*) deviceStartingPointIdx);
		printf("  Slave %d: deviceLimits: 0x%x\n", rank, (void*) deviceLimits);
#endif

	} else {
		maxBatchSize = getDefaultCPUBatchSize();
	}

#if DEBUG >= 2
	printf("  Slave %d: maxBatchSize = %d (%ld MB)\n", rank, maxBatchSize, maxBatchSize*sizeof(RESULT_TYPE) / (1024 * 1024));
#endif

	//while (parameters->remote) {				// TODO: Swap these for remote configuration
	while(true){

		// Send 'ready' signal to master
		#if DEBUG >= 2
		printf("  Slave %d: Sending READY...\n", rank);
		#endif
		MPI_Send(nullptr, 0, MPI_INT, 0, TAG_READY, comm);
		MPI_Send(&maxBatchSize, 1, MPI_UNSIGNED_LONG, 0, TAG_MAX_DATA_COUNT, comm);

		// Receive data (length and starting point) to compute
		#if DEBUG >= 2
		printf("  Slave %d: Waiting for data...\n", rank);
		#endif
		MPI_Recv(&numOfElements, 1, MPI_INT, 0, TAG_DATA_COUNT, comm, &status);
		MPI_Recv(startPointIdx, parameters->D, MPI_UNSIGNED_LONG, 0, TAG_DATA, comm, &status);

		// If received more data...
		if (numOfElements > 0) {
			#if DEBUG >= 1
			printf("  Slave %d: Running for %d elements...\n", rank, numOfElements);
			#endif
			#if DEBUG >= 3
			printf("  Slave %d: Got %d elements starting from  ", rank, numOfElements);
			for (unsigned int i = 0; i < parameters->D; i++)
				cout << startPointIdx[i] << " ";
			cout << endl;
			#endif
			fflush(stdout);

			// If batchSize was increased, allocate more memory for the results
			if (allocatedElements < numOfElements) {
				#if DEBUG >= 2
				printf("  Slave %d: Allocating more memory (%d -> %d elements, %ld MB)\n", rank, allocatedElements, numOfElements, (numOfElements*sizeof(RESULT_TYPE)) / (1024 * 1024));
				fflush(stdout);
				#endif
				allocatedElements = numOfElements;

				tmpResults = (RESULT_TYPE*)realloc(tmpResults, allocatedElements * sizeof(RESULT_TYPE));

				// If we are on GPU, we need to allocate more memory there as well
				if (gpuId > -1) {
					cudaFree(deviceResults);
					cce();

					cudaMalloc(&deviceResults, allocatedElements * sizeof(RESULT_TYPE));
					cce();
				}

				#if DEBUG >=2

				printf("  Slave %d: tmpResults = 0x%x\n", rank, tmpResults);
				if(gpuId > -1)
					printf("  Slave %d: deviceResults = 0x%x\n", rank, deviceResults);

				#endif
			}

			// Calculate the results
			if (gpuId > -1) {

				// Copy starting point indices to device
				cudaMemcpy(deviceStartingPointIdx, startPointIdx, parameters->D * sizeof(unsigned long), cudaMemcpyHostToDevice);
				cce();

				// Call the kernel
				blockSize = BLOCK_SIZE < numOfElements ? BLOCK_SIZE : numOfElements;
				numOfBlocks = (numOfElements + blockSize - 1) / blockSize;
				validate_kernel<ImplementedModel><<<numOfBlocks, blockSize>>>(deviceModelAddress, deviceStartingPointIdx, deviceResults, deviceLimits, parameters->D, numOfElements);
				cce();

				// Wait for kernel to finish
				cudaDeviceSynchronize();
				cce();

				if(! parameters->benchmark){
					// Get results from device
					cudaMemcpy(tmpResults, deviceResults, numOfElements * sizeof(RESULT_TYPE), cudaMemcpyDeviceToHost);
					cce();
				}

			} else {

				cpu_kernel<ImplementedModel>(startPointIdx, tmpResults, limits, parameters->D, numOfElements);

			}

			// Send the results to master
			#if DEBUG >= 2
			printf("  Slave %d: Sending %d RESULTS...\n", rank, numOfElements);
			#endif
			#if DEBUG >= 4
			// Print results
			printf("  Slave %d results: ", rank);
			for (int i = 0; i < numOfElements; i++) {
				printf("%f ", tmpResults[i]);
			}
			printf("\n");
			#endif
			MPI_Send(tmpResults, parameters->benchmark ? 0 : numOfElements, RESULT_MPI_TYPE, 0, TAG_RESULTS, comm);

		} else {
			// No more data
			break;
		}
	}

	// Notify master about exiting
	MPI_Send(nullptr, 0, MPI_INT, 0, TAG_EXITING, comm);

	// Finalize GPU
	if (gpuId > -1) {
		// Delete the model object on the device
		delete_model_kernel<ImplementedModel><<<1, 1 >>>(deviceModelAddress);
		cce();

		// Free the space for the model's address on the device
		cudaFree(deviceModelAddress);		cce();
		cudaFree(deviceResults);			cce();
		cudaFree(deviceStartingPointIdx);	cce();
		cudaFree(deviceLimits);				cce();
	}

	delete[] startPointIdx;
	free(tmpResults);
}

#endif
