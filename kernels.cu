#ifndef KERNELS_CU
#define KERNELS_CU

#include <cuda.h>
#include <omp.h>

#include "utilities.h"

using namespace std;

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
__global__ void validate_kernel(ImplementedModel** model, unsigned long* startingPointIdx, RESULT_TYPE* results, Limit* limits, unsigned int D, unsigned int numOfElements, unsigned int offset, void* dataPtr, int* listIndexPtr, int computeBatchSize) {
	unsigned int threadStart = offset + (((blockIdx.x * blockDim.x) + threadIdx.x) * computeBatchSize);
	unsigned int end = min(offset + numOfElements, threadStart + computeBatchSize);

	DATA_TYPE point[MAX_DIMENSIONS];
	unsigned long currentIndex[MAX_DIMENSIONS];
	unsigned long carry;
	unsigned int d;

	// Load data on shared memory
	__shared__ unsigned long shStartingPointIdx[MAX_DIMENSIONS];
	__shared__ DATA_TYPE shStep[MAX_DIMENSIONS];
	__shared__ unsigned int shN[MAX_DIMENSIONS];
	__shared__ DATA_TYPE shLowerLimit[MAX_DIMENSIONS];
	if(blockDim.x < D){
		// If threads are <D, thread 0 will load everything
		if(threadIdx.x == 0){
			for(d=0 ; d<D; d++){
				shStartingPointIdx[d] = startingPointIdx[d];
				shStep[d] = limits[d].step;
				shN[d] = limits[d].N;
				shLowerLimit[d] = limits[d].lowerLimit;
			}
		}
	}else if(threadIdx.x < D){
		shStartingPointIdx[threadIdx.x] = startingPointIdx[threadIdx.x];
		shStep[threadIdx.x] = limits[threadIdx.x].step;
		shN[threadIdx.x] = limits[threadIdx.x].N;
		shLowerLimit[threadIdx.x] = limits[threadIdx.x].lowerLimit;
	}
	__syncthreads();

	// Initialize currentIndex and point
	carry = threadStart;
	for (d = 0; d < D; d++) {
		currentIndex[d] = (shStartingPointIdx[d] + carry) % shN[d];
		carry = (shStartingPointIdx[d] + carry) / shN[d];

		// Calculate the exact coordinate i
		point[d] = shLowerLimit[d] + currentIndex[d] * shStep[d];
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
				carry = atomicAdd(listIndexPtr, D);
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

				// point[d] += limits[d].step is also an option
				point[d] = shLowerLimit[d] + shStep[d] * currentIndex[d];
				break;
			}else{
				// This dimension overflowed, initialize it and increment the next one
				currentIndex[d] = 0;
				point[d] = shLowerLimit[d];
				d++;
			}
		}

		threadStart++;
	}
}

// CPU kernel to run the computation
template<class ImplementedModel>
void cpu_kernel(unsigned long* startingPointIdx, RESULT_TYPE* results, Limit* limits, unsigned int D, int numOfElements, void* dataPtr, int* listIndexPtr) {
	ImplementedModel model = ImplementedModel();

	omp_set_nested(1);		// We are already in a parallel region since slaveProcess()
	#pragma omp parallel
	{
		DATA_TYPE* point = new DATA_TYPE[D];
		unsigned long* currentIndex = new unsigned long[D];
		unsigned long carry;
		unsigned int d, processed, localNumOfElements, elementsPerThread, start, end;

		// Calculate start and end
		elementsPerThread = numOfElements / omp_get_num_threads();
		start = omp_get_thread_num()*elementsPerThread;
		end = start + elementsPerThread;
		if(omp_get_thread_num() == omp_get_num_threads()-1){
			end = numOfElements;
		}
		localNumOfElements = end - start;

		// Initialize currentIndex and point
		carry = start;
		for (d = 0; d < D; d++) {
			currentIndex[d] = (startingPointIdx[d] + carry) % limits[d].N;
			carry = (startingPointIdx[d] + carry) / limits[d].N;

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

					// point[d] += limits[d].step is also an option
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
