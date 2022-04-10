#include "utilities.h"

#include "definitions.h"

namespace desf {

unsigned long getMaxCPUBytes(){
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return (pages * page_size)/4;
}
unsigned long getMaxGPUBytes(){
    int gpus;
    unsigned long minBytes = ULONG_MAX;

    // Get GPUs count
    cudaGetDeviceCount(&gpus);

    for(int i=0; i<gpus; i++){
        // Calculate and keep the minimum bytes
        minBytes = std::min(minBytes, getMaxGPUBytesForGpu(i));
    }

    return minBytes;
}

unsigned long getMaxGPUBytesForGpu(int id){
    size_t freeMem, totalMem;				// Bytes of free,total memory on GPU

    cudaSetDevice(id);
    cudaMemGetInfo(&freeMem, &totalMem);

    return (unsigned long) (freeMem - MEM_GPU_SPARE_BYTES);
}

int getCpuStats(float* uptime, float* idleTime){
    FILE *fp;

    fp = fopen ("/proc/uptime", "r");
    if (fp != NULL){
        float localUptime;
        float localIdleTime;

        if(fscanf(fp, "%f %f", &localUptime, &localIdleTime) == 2){
            *uptime = localUptime;
            *idleTime = localIdleTime / get_nprocs_conf();

            fclose (fp);
            return 0;
        } else {
            fclose (fp);
            return -2;
        }
    } else {
        return -1;
    }
}

#define RECV_SLEEP_US 100     // Time in micro-seconds to sleep between checking for data in MPI_Recv
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

}
