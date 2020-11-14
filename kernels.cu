#ifndef KERNELS_CU
#define KERNELS_CU

#include <cuda.h>
#include <omp.h>

#include "utilities.h"

using namespace std;

#define MAX_CONSTANT_MEMORY (65536 - 24)			// Don't know why...
__device__ __constant__ char constantMemoryPtr[MAX_CONSTANT_MEMORY];

template<class ImplementedModel>
void cudaMemcpyToSymbolWrapper(const void* src, size_t count, size_t offset){
	cudaMemcpyToSymbol(constantMemoryPtr, src, count, offset, cudaMemcpyHostToDevice);
}

// CUDA kernel to create the 'Model' object on device
template<class ImplementedModel>
__global__ void create_model_kernel(ImplementedModel** deviceModelAddress) {
	(*deviceModelAddress) = new ImplementedModel();
}

// CUDA kernel to delete the 'Model' object on device
template<class ImplementedModel>
__global__ void delete_model_kernel(ImplementedModel** deviceModelAddress) {
	delete (*deviceModelAddress);
}

// CUDA kernel to run the computation
template<class ImplementedModel>
__global__ void validate_kernel(ImplementedModel** model, RESULT_TYPE* results, unsigned long startingPointLinearIndex,
	const unsigned int D, const unsigned long numOfElements, const unsigned long offset, void* dataPtr,
	int dataSize, bool useSharedMemoryForData, bool useConstantMemoryForData, int* listIndexPtr,
	const int computeBatchSize) {

	unsigned long threadStart = offset + (((blockIdx.x * blockDim.x) + threadIdx.x) * computeBatchSize);
	unsigned long end = min(offset + numOfElements, threadStart + computeBatchSize);

	int d;
	unsigned long tmp, remainder;

	// Constant Memory
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
		threadIdx.x * sizeof(unsigned int) 			// Bypass one element for each previous threads
	];

	remainder = threadStart + startingPointLinearIndex;
	for (d = D-1; d>=0; d--){
		tmp = idxSteps[d];

		currentIndex[blockDim.x * d] = remainder / tmp;
		remainder -= currentIndex[blockDim.x * d] * tmp;

		// Calculate the exact coordinate i
		point[d] = limits[d].lowerLimit + currentIndex[blockDim.x * d] * limits[d].step;
	}

	while(threadStart < end){
		// Evaluate point
		if(listIndexPtr == nullptr){
			// We are running as SAVE_TYPE_ALL
			// Run the validation function and save the result to the global memory
			results[threadStart] = (*model)->validate_gpu(point, dataPtr);
		}else{
			// We are running as SAVE_TYPE_LIST
			// Run the validation function and pass its result to toBool
			if((*model)->toBool((*model)->validate_gpu(point, dataPtr))){
				// Append element to the list
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
			currentIndex[blockDim.x * d]++;

			if(currentIndex[blockDim.x * d] < limits[d].N){
				// No need to recalculate the rest of the dimensions

				// point[d] += limits[d].step; // is also an option
				point[d] = limits[d].lowerLimit + limits[d].step * currentIndex[blockDim.x * d];
				break;
			}else{
				// This dimension overflowed, initialize it and increment the next one
				currentIndex[blockDim.x * d] = 0;
				point[d] = limits[d].lowerLimit;
				d++;
			}
		}

		threadStart++;
	}
}

// CPU kernel to run the computation
template<class ImplementedModel>
void cpu_kernel(RESULT_TYPE* results, Limit* limits, unsigned int D, unsigned long numOfElements, void* dataPtr, int* listIndexPtr,
	unsigned long long* idxSteps, unsigned long startingPointLinearIndex) {
	ImplementedModel model = ImplementedModel();

	omp_set_nested(1);		// We are already in a parallel region since slaveProcess()
	#pragma omp parallel
	{
		DATA_TYPE* point = new DATA_TYPE[D];
		unsigned long* currentIndex = new unsigned long[D];
		unsigned long carry, processed, localNumOfElements, elementsPerThread, start, end;
		int d;

		// Calculate start and end
		elementsPerThread = numOfElements / omp_get_num_threads();
		start = omp_get_thread_num()*elementsPerThread;
		end = start + elementsPerThread;
		if(omp_get_thread_num() == omp_get_num_threads()-1){
			end = numOfElements;
		}
		localNumOfElements = end - start;

		// Initialize currentIndex and point
		long newIndex, remainder;
		remainder = start + startingPointLinearIndex;
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
				results[start + processed] = model.validate_cpu(point, dataPtr);
			}else{
				// We are running as SAVE_TYPE_LIST
				// Run the validation function and pass its result to toBool
				if(model.toBool(model.validate_cpu(point, dataPtr))){
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

		delete[] point;
	}

}

#endif
