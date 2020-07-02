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
__global__ void validate_kernel(ImplementedModel** model, RESULT_TYPE* results, const Limit* limits, unsigned long startingPointLinearIndex,
	const unsigned int D, const unsigned long long* idxSteps, const unsigned int numOfElements, const unsigned int offset, void* dataPtr,
	int* listIndexPtr, const int computeBatchSize) {
	unsigned int threadStart = offset + (((blockIdx.x * blockDim.x) + threadIdx.x) * computeBatchSize);
	unsigned int end = min(offset + numOfElements, threadStart + computeBatchSize);

	DATA_TYPE point[MAX_DIMENSIONS];
	unsigned int currentIndex[MAX_DIMENSIONS];
	int d;
	unsigned long tmp, remainder;
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
			currentIndex[d]++;

			if(currentIndex[d] < limits[d].N){
				// No need to recalculate the rest of the dimensions

				point[d] += limits[d].step; // is also an option
				// point[d] = limits[d].lowerLimit + limits[d].step * currentIndex[d];
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
template<class ImplementedModel>
void cpu_kernel(RESULT_TYPE* results, Limit* limits, unsigned int D, int numOfElements, void* dataPtr, int* listIndexPtr,
	unsigned long long* idxSteps, unsigned long startingPointLinearIndex) {
	ImplementedModel model = ImplementedModel();

	omp_set_nested(1);		// We are already in a parallel region since slaveProcess()
	#pragma omp parallel
	{
		DATA_TYPE* point = new DATA_TYPE[D];
		unsigned long* currentIndex = new unsigned long[D];
		unsigned long carry;
		int d, processed, localNumOfElements, elementsPerThread, start, end;

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

					point[d] += limits[d].step; // is also an option
					// point[d] = limits[d].lowerLimit + limits[d].step * currentIndex[d];
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
