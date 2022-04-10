#pragma once

#include "types.h"

// CPU kernel to run the computation
void cpu_kernel(desf::validationFunc_t validationFunc, desf::toBool_t toBool,
                RESULT_TYPE* results, const desf::Limit* limits, unsigned int D, unsigned long numOfElements,
                void* dataPtr, int* listIndexPtr, const unsigned long long* idxSteps, unsigned long startingPointLinearIndex,
                bool dynamicScheduling, int batchSize);
