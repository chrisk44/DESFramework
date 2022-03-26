#pragma once

#include "utilities/utilities.h"

// CPU kernel to run the computation
void cpu_kernel(validationFunc_t validationFunc, toBool_t toBool,
                RESULT_TYPE* results, Limit* limits, unsigned int D, unsigned long numOfElements,
                void* dataPtr, int* listIndexPtr, unsigned long long* idxSteps, unsigned long startingPointLinearIndex,
                bool dynamicScheduling, int batchSize);
