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
		usleep(RECV_SLEEP_MS * 1000);
		MPI_Test(&request, &flag, status);
	}
}

void Stopwatch::start(){
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    started = true;
}
void Stopwatch::stop(){
    if(!started){
        printf("[E] -----------(stop())------------- CLOCK HAS NOT BEEN STARTED --------------------------------------\n");
    }else if(stopped){
        printf("[E] -----------(stop())--------------- CLOCK ALREADY STOPPED -----------------------------------------\n");
    }
    fflush(stdout);

    stopped = true;
    bool flag = false;
    do{
        if(flag){
            printf("---------------------- looped because of negative clock\n");
        }
        flag = true;
        clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
    }while(getMsec() < 0);  // Clock may backwards for some reason

}
void Stopwatch::reset(){
    started = false;
    stopped = false;
}
float Stopwatch::getNsec(){
    if(!started){
        printf("[E] -----------(getNsec())------------- CLOCK HAS NOT BEEN STARTED --------------------------------------\n");
    }else if(!stopped){
        printf("[E] -----------(getNsec())------------- CLOCK HAS NOT BEEN STOPPED --------------------------------------\n");
    }
    timespec difference;
	if ((t2.tv_nsec - t1.tv_nsec)<0) {
		difference.tv_sec = t2.tv_sec - t1.tv_sec - 1;
		difference.tv_nsec = 1000000000 + t2.tv_nsec - t1.tv_nsec;
	} else {
		difference.tv_sec = t2.tv_sec - t1.tv_sec;
		difference.tv_nsec = t2.tv_nsec - t1.tv_nsec;
	}

    // printf("\nt1 is %d s and %d nsec\n", t1.tv_sec, t1.tv_nsec);
    // printf("t2 is %d s and %d nsec\n", t2.tv_sec, t2.tv_nsec);
    // printf("difference is %d s and %d nsec\n\n", difference.tv_sec, difference.tv_nsec);
    return difference.tv_sec * 1000000000 + difference.tv_nsec;
}
float Stopwatch::getMsec(){
    return getNsec() / 1000000.0;
}
