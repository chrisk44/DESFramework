#ifndef KERNELS_H
#define KERNELS_H

#include "utilities.h"

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
__global__ void validate_kernel(ImplementedModel** model, long* startingPointIdx, bool* results, Limit* limits, unsigned int D) {
	float point[MAX_DIMENSIONS];
	long tmpIndex;

	// Calculate 'myIndex = startingPointIdx + threadIdx.x' and then the exact point
	unsigned int i;
	unsigned int carry = threadIdx.x;
	for (i = 0; i < D; i++) {
		tmpIndex = (startingPointIdx[i] + carry) % limits[i].N;
		carry = (startingPointIdx[i] + carry) / limits[i].N;

		// Calculate the exact coordinate i
		point[i] = limits[i].lowerLimit + tmpIndex * abs(limits[i].lowerLimit - limits[i].upperLimit) / limits[i].N;
	}

	// Run the validation function and save the result to the global memory
	results[threadIdx.x] = (*model)->validate_gpu(point);
}

// CPU kernel to run the computation
template<class ImplementedModel>
void cpu_kernel(long* startingPointIdx, bool* results, Limit* limits, unsigned int D, long numOfElements) {
	float* point = new float[D];
	long tmpIndex;

	ImplementedModel model = ImplementedModel();

	for (long j = 0; j < numOfElements; j++) {
		// Calculate 'myIndex = startingPointIdx + j' and then the exact point
		unsigned int i;
		unsigned int carry = j;
		for (i = 0; i < D; i++) {
			tmpIndex = (startingPointIdx[i] + carry) % limits[i].N;
			carry = (startingPointIdx[i] + carry) / limits[i].N;

			// Calculate the exact coordinate i
			point[i] = limits[i].lowerLimit + tmpIndex * abs(limits[i].lowerLimit - limits[i].upperLimit) / limits[i].N;
		}

		// Run the validation function
		results[j] = model.validate_cpu(point);
	}

	delete[] point;
}

#endif
