#include <omp.h>

#include "cpuKernel.h"

// CPU kernel to run the computation
void cpu_kernel(validationFunc_t validationFunc, toBool_t toBool, RESULT_TYPE* results, Limit* limits, unsigned int D, unsigned long numOfElements, void* dataPtr, int* listIndexPtr,
    unsigned long long* idxSteps, unsigned long startingPointLinearIndex, bool dynamicScheduling, int batchSize) {

    unsigned long currentBatchStart = startingPointLinearIndex;
    unsigned long globalLast = startingPointLinearIndex + numOfElements - 1;

    omp_set_nested(1);		// We are already in a parallel region since slaveProcess()
    #pragma omp parallel shared(currentBatchStart)
    {
        DATA_TYPE point[D];
        unsigned long* currentIndex = new unsigned long[D];
        unsigned long carry, processed, localNumOfElements, elementsPerThread, start, lastElement;
        int d;

        // Adjust for small workloads
        if(batchSize > numOfElements/omp_get_num_threads() && numOfElements >= omp_get_num_threads()){
            batchSize = numOfElements/omp_get_num_threads();
        }

        // Calculate start and end for static scheduling
        elementsPerThread = numOfElements / omp_get_num_threads();
        start = startingPointLinearIndex + omp_get_thread_num()*elementsPerThread;
        lastElement = start + elementsPerThread - 1;
        if(omp_get_thread_num() == omp_get_num_threads()-1){
            lastElement = globalLast;
        }

        while(true){
            if(dynamicScheduling){
                start = __sync_fetch_and_add(&currentBatchStart, batchSize);
                if(start > globalLast)
                    break;

                lastElement = std::min(start + batchSize - 1, globalLast);
            }

            localNumOfElements = lastElement - start + 1;

            // Initialize currentIndex and point
            long newIndex, remainder;
            remainder = start;
            for (d = D-1; d>=0; d--){

                newIndex = remainder / idxSteps[d];
                currentIndex[d] = newIndex;
                remainder -= newIndex*idxSteps[d];

                // Calculate the exact coordinate i
                point[d] = limits[d].lowerLimit + currentIndex[d] * limits[d].step;
            }

            processed = 0;
            while(processed < localNumOfElements){
                // Evaluate point
                if(listIndexPtr == nullptr){
                    // We are running as SAVE_TYPE_ALL
                    // Run the validation function and save the result to the global memory
                    results[start + processed] = validationFunc(point, dataPtr);
                }else{
                    // We are running as SAVE_TYPE_LIST
                    // Run the validation function and pass its result to toBool
                    if(toBool(validationFunc(point, dataPtr))){
                        // Append element to the list
                        carry = __sync_fetch_and_add(listIndexPtr, D);
                        for(d = 0; d < D; d++){
                            ((DATA_TYPE *)results)[carry + d] = point[d];
                        }
                    }
                }

                // Increment indices and point
                d = 0;
                while(d < D){
                    // Increment dimension d
                    currentIndex[d]++;

                    if(currentIndex[d] < limits[d].N){
                        // No need to recalculate the rest of the dimensions

                        // point[d] += limits[d].step; // is also an option
                        point[d] = limits[d].lowerLimit + limits[d].step * currentIndex[d];
                        break;
                    }else{
                        // This dimension overflowed, initialize it and increment the next one
                        currentIndex[d] = 0;
                        point[d] = limits[d].lowerLimit;
                        d++;
                    }
                }

                processed++;
            }

            if(!dynamicScheduling){
                break;
            }
        }
    }
}
