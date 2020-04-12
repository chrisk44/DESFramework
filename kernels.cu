#ifndef KERNELS_CU
#define KERNELS_CU

#include <cuda.h>
#include <omp.h>

#include "utilities.h"

#include <iostream>
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
__global__ void validate_kernel(ImplementedModel** model, unsigned long* startingPointIdx, RESULT_TYPE* results, Limit* limits, unsigned int D, unsigned int numOfElements) {
	unsigned int threadX = (blockIdx.x * BLOCK_SIZE) + threadIdx.x;
	if (threadX < numOfElements) {
		DATA_TYPE point[MAX_DIMENSIONS];
		unsigned long tmpIndex, carry;
		unsigned int i;
		/*DATA_TYPE step[MAX_DIMENSINS];
		for (i = 0; i < D; i++) {
			step[i] = abs(limits[i].lowerLimit - limits[i].upperLimit) / limits[i].N;
		}*/

		// Calculate 'myIndex = startingPointIdx + threadIdx.x' and then the exact point
		carry = threadX;
		for (i = 0; i < D; i++) {
			tmpIndex = (startingPointIdx[i] + carry) % limits[i].N;
			carry = (startingPointIdx[i] + carry) / limits[i].N;

			// Calculate the exact coordinate i
			point[i] = limits[i].lowerLimit + tmpIndex * (abs(limits[i].lowerLimit - limits[i].upperLimit) / limits[i].N);
		}

		// Run the validation function and save the result to the global memory
		results[threadX] = (*model)->validate_gpu(point);
	}
}

// CPU kernel to run the computation
template<class ImplementedModel>
void cpu_kernel(unsigned long* startingPointIdx, RESULT_TYPE* results, Limit* limits, unsigned int D, int numOfElements) {
	DATA_TYPE* step = new DATA_TYPE[D];
	ImplementedModel model = ImplementedModel();

	for (unsigned int i = 0; i < D; i++) {
		step[i] = abs(limits[i].lowerLimit - limits[i].upperLimit) / limits[i].N;
	}

	// TODO: Change constant num_thread(4)
	// TODO: No performance improvement
	#pragma omp parallel shared(step, model, startingPointIdx, results, limits, D, numOfElements) num_threads(4)
	{
		DATA_TYPE* point = new DATA_TYPE[D];
		unsigned long tmpIndex, carry;
		unsigned int i, j;

		for (j = omp_get_thread_num(); j < numOfElements; j+=omp_get_num_threads()) {
			// Calculate 'myIndex = startingPointIdx + j' and then the exact point
			carry = j;

			for (i = 0; i < D; i++) {
				tmpIndex = (startingPointIdx[i] + carry) % limits[i].N;
				carry = (startingPointIdx[i] + carry) / limits[i].N;

				// Calculate the exact coordinate i
				point[i] = limits[i].lowerLimit + tmpIndex * step[i];
			}

			// Run the validation function
			results[j] = model.validate_cpu(point);
		}

		#if DEBUG >=4
			//cout << "Point (" << point[0] << "," << point[1] << ") returned " << results[j] << endl;
		#endif

		delete[] point;
	}

}

#endif
