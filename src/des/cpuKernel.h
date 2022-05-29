#pragma once

#include <functional>

#include "types.h"

// CPU kernel to run the computation
void cpu_kernel(desf::validationFunc_t validationFunc, desf::toBool_t toBool,
                unsigned int D, const desf::Limit* limits, const unsigned long long* idxSteps,
                unsigned long startingPointLinearIndex, unsigned long numOfElements,
                RESULT_TYPE* allResults, std::function<void(size_t)> addListResult,
                void* dataPtr, bool dynamicScheduling, int batchSize);
