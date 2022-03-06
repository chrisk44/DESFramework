#include <cuda.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "kernels.h"
#include "utilities.h"

using namespace std;

void cudaMemcpyToSymbolWrapper(const void* src, size_t count, size_t offset){
	cudaMemcpyToSymbol(constantMemoryPtr, src, count, offset, cudaMemcpyHostToDevice);
}

// CUDA kernel to run the computation
__global__ void validate_kernel(validationFunc_t validationFunc, toBool_t toBool, RESULT_TYPE* results, unsigned long startingPointLinearIndex,
	const unsigned int D, const unsigned long numOfElements, const unsigned long offset, void* dataPtr,
	int dataSize, bool useSharedMemoryForData, bool useConstantMemoryForData, int* listIndexPtr,
	const int computeBatchSize) {

	unsigned long threadStart = offset + ((((unsigned long) blockIdx.x * blockDim.x) + threadIdx.x) * computeBatchSize);
	unsigned long end = min(offset + numOfElements, threadStart + computeBatchSize);

	int d;
	unsigned long tmp, remainder;

	// Constant Memory layout is: Limits[], idxSteps[], ?Model-data
	Limit* limits = (Limit*) constantMemoryPtr;
	unsigned long long* idxSteps = (unsigned long long*) &constantMemoryPtr[D*sizeof(Limit)];

	if(useConstantMemoryForData)
		dataPtr = (void*) &constantMemoryPtr[D * (sizeof(Limit) + sizeof(unsigned long long))];

	// Shared memory layout is: point(thread0)[], point(thread1)[], ..., indices (with stride)..., ?Model-data
	extern __shared__ char sharedMem[];
	DATA_TYPE *point;
	unsigned int *currentIndex;

	// If we are using shared memory for the model's data, fetch them
	if(useSharedMemoryForData){
		char* sharedDataPtr = &sharedMem[blockDim.x * D * (sizeof(DATA_TYPE) + sizeof(unsigned int))];
		// If we have enough threads in the block, use them to bring the data simultaneously
		if(blockDim.x >= dataSize){
			if(threadIdx.x < dataSize){
				sharedDataPtr[threadIdx.x] = ((char*)dataPtr)[threadIdx.x];
			}
		}else{										// TODO: IMPROVEMENT: Use multiple threads where each one loads more bytes of data (useful when the threads are less than dataSize)
			if(threadIdx.x == 0){
				for(int d=0; d<dataSize; d++){
					sharedDataPtr[d] = ((char*)dataPtr)[d];
				}
			}
		}

		dataPtr = sharedDataPtr;

		__syncthreads();
	}

	point = (DATA_TYPE*) &sharedMem[
		threadIdx.x * D * sizeof(DATA_TYPE)			// Bypass the previous threads' points
	];

	currentIndex = (unsigned int*) &sharedMem[
		blockDim.x * D * sizeof(DATA_TYPE) +		// Bypass all threads' points
		threadIdx.x * D * sizeof(unsigned int) 		// Bypass the previous threads' indices
	];

	remainder = threadStart + startingPointLinearIndex;
	for (d = D-1; d>=0; d--){
		tmp = idxSteps[d];

		currentIndex[d] = remainder / tmp;
		remainder -= currentIndex[d] * tmp;

		// Calculate the exact coordinate i
		point[d] = limits[d].lowerLimit + currentIndex[d] * limits[d].step;
	}

	while(threadStart < end){
		// Evaluate point
		if(listIndexPtr == nullptr){
			// We are running as SAVE_TYPE_ALL
			// Run the validation function and save the result to the global memory
			results[threadStart] = validationFunc(point, dataPtr);
		}else{
			// We are running as SAVE_TYPE_LIST
			// Run the validation function and pass its result to toBool
			if(toBool(validationFunc(point, dataPtr))){
				// Append element to the list
				// TODO: STABILITY: Handle overflow
				tmp = atomicAdd(listIndexPtr, D);
				for(d = 0; d < D; d++){
					((DATA_TYPE *)results)[tmp + d] = point[d];
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

		threadStart++;
	}
}

// CPU kernel to run the computation
void cpu_kernel(validationFunc_t validationFunc, toBool_t toBool, RESULT_TYPE* results, Limit* limits, unsigned int D, unsigned long numOfElements, void* dataPtr, int* listIndexPtr,
	unsigned long long* idxSteps, unsigned long startingPointLinearIndex, bool dynamicScheduling, int batchSize) {

	unsigned long currentBatchStart = startingPointLinearIndex;
	unsigned long globalLast = startingPointLinearIndex + numOfElements - 1;

	omp_set_nested(1);		// We are already in a parallel region since slaveProcess()
	#pragma omp parallel shared(currentBatchStart)
	{
		DATA_TYPE* point = new DATA_TYPE[D];
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

				lastElement = min(start + batchSize - 1, globalLast);
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

		delete[] point;
	}

}
