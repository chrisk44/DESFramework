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
	unsigned int threadOffset = ((blockIdx.x * BLOCK_SIZE) + threadIdx.x) * COMPUTE_BATCH_SIZE;
	unsigned int end = min(numOfElements, threadOffset + COMPUTE_BATCH_SIZE);

	DATA_TYPE point[MAX_DIMENSIONS];
	unsigned long tmpIndex, carry;
	unsigned int i, d;

	// Load data on registers
	// unsigned long shStartingPointIdx[MAX_DIMENSIONS];
	// DATA_TYPE shStep[MAX_DIMENSIONS];
	// unsigned int shN[MAX_DIMENSIONS];
	// DATA_TYPE shLowerLimit[MAX_DIMENSIONS];
	// for (d = 0; d < D; d++) {
	// 	shStartingPointIdx[d] = startingPointIdx[d];
	// 	shStep[d] = limits[d].step;
	// 	shN[d] = limits[d].N;
	// 	shLowerLimit[d] = limits[d].lowerLimit;
	// }

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


	for(i=threadOffset; i<end; i++){
		// Calculate point for (startingPointIdx + threadOffset + i)
		carry = i;
		for (d = 0; d < D; d++) {
			tmpIndex = (shStartingPointIdx[d] + carry) % shN[d];
			carry = (shStartingPointIdx[d] + carry) / shN[d];

			// Calculate the exact coordinate i
			point[d] = shLowerLimit[d] + tmpIndex * shStep[d];//* step[i];
		}

		// Run the validation function and save the result to the global memory
		results[i] = (*model)->validate_gpu(point);
	}
}

// CPU kernel to run the computation
template<class ImplementedModel>
void cpu_kernel(unsigned long* startingPointIdx, RESULT_TYPE* results, Limit* limits, unsigned int D, int numOfElements) {
	ImplementedModel model = ImplementedModel();

	omp_set_nested(1);		// We are already in a parallel region since slaveProcess()
	#pragma omp parallel
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
				point[i] = limits[i].lowerLimit + tmpIndex * limits[i].step;
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
