#pragma once

#define cce() {                                          \
  cudaError_t e = cudaGetLastError();                    \
  if (e != cudaSuccess) {                                \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                      \
    exit(1);                                             \
  }                                                      \
}

#define fatal(e){                                        \
    printf("Error: %s:%d: %s\n", __FILE__, __LINE__, e); \
    exit(3);                                             \
}

// Memory parameters
#ifndef MEM_GPU_SPARE_BYTES
    #define MEM_GPU_SPARE_BYTES 100*1024*1024
#endif

// Computing parameters
#define MAX_DIMENSIONS 30

#ifndef RESULT_TYPE
    #define RESULT_TYPE float
#endif
#ifndef DATA_TYPE
    #define DATA_TYPE double
#endif
#ifndef RESULT_MPI_TYPE
    #define RESULT_MPI_TYPE MPI_FLOAT
#endif
#ifndef DATA_MPI_TYPE
    #define DATA_MPI_TYPE MPI_DOUBLE
#endif

// Debugging
// #define DBG_START_STOP      // Messages about starting/stopping processes and threads
// #define DBG_QUEUE           // Messages about queueing work (coordinator->worker threads, worker->gpu streams)
// #define DBG_MPI_STEPS       // Messages after each MPI step
// #define DBG_RATIO           // Messages about changes in ratios (masterProcess and coordinatorThread)
// #define DBG_DATA            // Messages about the exact data being assigned (start points)
 #define DBG_MEMORY          // Messages about memory management (addresses, reallocations)
// #define DBG_RESULTS         // Messages with the exact results being passed around
// #define DBG_TIME            // Print time measuraments for various parts of the code
#define DBG_SNH             // Should not happen

// MPI
#define RECV_SLEEP_US 100     // Time in micro-seconds to sleep between checking for data in MPI_Recv
#define TAG_READY 1
#define TAG_DATA 2
#define TAG_RESULTS 3
#define TAG_MAX_DATA_COUNT 4
#define TAG_EXITING 5
#define TAG_RESULTS_DATA 6
