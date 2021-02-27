#include "framework.h"

void ParallelFramework::slaveProcess(validationFunc_t validation_cpu, validationFunc_t validation_gpu, toBool_t toBool_cpu, toBool_t toBool_gpu) {
	/*******************************************************************
	********** Calculate number of worker threads (#GPUs + 1CPU) *******
	********************************************************************/
	int numOfThreads = 0;

	if(parameters->processingType != PROCESSING_TYPE_CPU)
		cudaGetDeviceCount(&numOfThreads);

	if(parameters->processingType != PROCESSING_TYPE_GPU)
		numOfThreads++;

	if(numOfThreads == 0){
		printf("[%d] SlaveProcess: Error: cudaGetDeviceCount returned 0\n", rank);
		return;
	}


	/*******************************************************************
	********************** Initialization ******************************
	********************************************************************/
	ThreadCommonData threadCommonData;
	sem_init(&threadCommonData.semResults, 0, 0);
	sem_init(&threadCommonData.semSync, 0, 1);
	threadCommonData.results = nullptr;

	ComputeThreadInfo* computeThreadInfo = new ComputeThreadInfo[numOfThreads];
	for(int i=0; i<numOfThreads; i++){
		sem_init(&computeThreadInfo[i].semStart, 0, 0);
		computeThreadInfo[i].ratio = (float)1/numOfThreads;
		computeThreadInfo[i].totalRatio = 0;
	}

	#ifdef DBG_START_STOP
		printf("[%d] SlaveProcess: Spawning %d worker threads...\n", rank, numOfThreads);
	#endif

	/*******************************************************************
	*************** Launch coordinator and worker threads **************
	********************************************************************/
	#pragma omp parallel num_threads(numOfThreads + 1) shared(computeThreadInfo) 	// +1 thread to handle the communication with masterProcess
	{
		int tid = omp_get_thread_num();
		if(tid == 0){
			coordinatorThread(computeThreadInfo, &threadCommonData, omp_get_num_threads()-1);
		}else{
			// Calculate id: -1 -> CPU, 0+ -> GPU[id]
			computeThreadInfo[tid-1].id = tid - (parameters->processingType == PROCESSING_TYPE_GPU ? 1 : 2);

			computeThread(computeThreadInfo[tid - 1], &threadCommonData, validation_cpu, validation_gpu, toBool_cpu, toBool_gpu);
		}
	}

	/*******************************************************************
	***************************** Finalize *****************************
	********************************************************************/
	sem_destroy(&threadCommonData.semResults);
	sem_destroy(&threadCommonData.semSync);
	for(int i=0; i<numOfThreads; i++){
		sem_destroy(&computeThreadInfo[i].semStart);
	}
	delete[] computeThreadInfo;
}
