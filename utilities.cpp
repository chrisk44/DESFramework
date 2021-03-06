#include "utilities.h"

using namespace std;

unsigned long getDefaultCPUBatchSize(){
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return (pages * page_size)/(4*sizeof(RESULT_TYPE));
}
unsigned long getDefaultGPUBatchSize(){
    size_t freeMem, totalMem;				// Bytes of free,total memory on GPU
    int gpus;
    int minBatchSize = INT_MAX;

    // Get GPUs count
    cudaGetDeviceCount(&gpus);

    for(int i=0; i<gpus; i++){
        // Select GPU i
        cudaSetDevice(i);

        // Read device's memory info
        cudaMemGetInfo(&freeMem, &totalMem);

        // Calculate and keep the minimum batch size
        minBatchSize = min(minBatchSize, (int) ((freeMem - MEM_GPU_SPARE_BYTES) / sizeof(RESULT_TYPE)));
    }
    return minBatchSize;
}

void MMPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status* status){
    // MPI_Recv(buf, count, datatype, source, tag, comm, status);
    // return;

	MPI_Request request;
	int flag = 0;
	MPI_Irecv(buf, count, datatype, source, tag, comm, &request);
	while(!flag){
		usleep(RECV_SLEEP_US);
		MPI_Test(&request, &flag, status);
	}
}

void Stopwatch::start(){
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
}
void Stopwatch::stop(){
    clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
}
float Stopwatch::getNsec(){
    timespec difference;
	if ((t2.tv_nsec - t1.tv_nsec)<0) {
		difference.tv_sec = t2.tv_sec - t1.tv_sec - 1;
		difference.tv_nsec = 1000000000 + t2.tv_nsec - t1.tv_nsec;
	} else {
		difference.tv_sec = t2.tv_sec - t1.tv_sec;
		difference.tv_nsec = t2.tv_nsec - t1.tv_nsec;
	}

    return difference.tv_sec * 1000000000 + difference.tv_nsec;
}
float Stopwatch::getMsec(){
    return getNsec() / 1000000.0;
}
