#pragma once

#include <map>
#include <string>

#define cce() {                                          \
  cudaError_t e = cudaGetLastError();                    \
  if (e != cudaSuccess) {                                \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                      \
    exit(1);                                             \
  }                                                      \
}

#define fatal(e){                                        \
    std::string err = "DES Error: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + e; \
    throw std::runtime_error(err); \
}

// Memory parameters
#ifndef MEM_GPU_SPARE_BYTES
    #define MEM_GPU_SPARE_BYTES 500*1024*1024
#endif

#ifndef LOG_BUFFER_SIZE
    #define LOG_BUFFER_SIZE 65536
#endif

#ifndef RESULT_TYPE
    #define RESULT_TYPE float
    #define RESULT_MPI_TYPE MPI_FLOAT
#endif

#ifndef DATA_TYPE
    #define DATA_TYPE double
    #define DATA_MPI_TYPE MPI_DOUBLE
#endif

// Debugging
// #define DBG_START_STOP      // Messages about starting/stopping processes and threads
// #define DBG_QUEUE           // Messages about queueing work (coordinator->worker threads, worker->gpu streams)
// #define DBG_MPI_STEPS       // Messages after each MPI step
// #define DBG_RATIO           // Messages about changes in ratios (masterProcess and coordinatorThread)
// #define DBG_DATA            // Messages about the exact data being assigned (start points)
// #define DBG_MEMORY          // Messages about memory management (addresses, reallocations)
// #define DBG_RESULTS_RAW     // Messages with the exact results being passed around
// #define DBG_RESULTS         // Messages about the results
// #define DBG_TIME            // Print time measuraments for various parts of the code
// #define DBG_SCHEDULE        // Messages by schedulers
#define DBG_SNH             // Should not happen

// MPI
#define TAG_READY 1
#define TAG_DATA 2
#define TAG_RESULTS 3
#define TAG_MAX_DATA_COUNT 4
#define TAG_EXITING 5
#define TAG_RESULTS_DATA 6
#define TAG_RESULTS_COUNT 7

static const std::map<int, std::string> TAG_NAMES {
    { 1, "Ready" },
    { 2, "Data" },
    { 3, "Results" },
    { 4, "Max data count" },
    { 5, "Exiting" },
    { 6, "Results (data)" },
    { 7, "Results (count)" },
};
