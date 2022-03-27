#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <nvml.h>
#include <limits.h>
#include <mpi.h>
#include <mutex>
#include <semaphore.h>
#include <string>
#include <sys/sysinfo.h>
#include <unistd.h>

unsigned long getMaxCPUBytes();
unsigned long getMaxGPUBytes();
unsigned long getMaxGPUBytesForGpu(int id);
int getCpuStats(float* uptime, float* idleTime);

// MPI_Recv without busy wait
void MMPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status* status);
