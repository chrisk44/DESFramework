#include <limits.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <mpi.h>

#include "framework.h"

using namespace std;

ParallelFramework::ParallelFramework(bool initMPI) {
	// The user might want to do the MPI Initialization. Useful when the framework is used more than once in a program.
	if(initMPI){
		// Initialize MPI
		MPI_Init(nullptr, nullptr);
	}
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	valid = false;
}

void ParallelFramework::init(const std::vector<Limit>& _limits, const ParallelFrameworkParameters& _parameters){
	unsigned int i;

    parameters = _parameters;
	limits = _limits;

	// Verify parameters
    if (parameters.D == 0 || parameters.D>MAX_DIMENSIONS) {
		cout << "[Init] Error: Dimension must be between 1 and " << MAX_DIMENSIONS << endl;
		return;
	}

    for (i = 0; i < parameters.D; i++) {
		if (limits[i].lowerLimit > limits[i].upperLimit) {
			cout << "[Init] Error: Limits for dimension " << i << ": Lower limit can't be higher than upper limit" << endl;
			return;
		}

		if (limits[i].N == 0) {
			cout << "[Init] Error: Limits for dimension " << i << ": N must be > 0" << endl;
			return;
		}
	}

    if(parameters.dataPtr == nullptr && parameters.dataSize > 0){
		cout << "[Init] Error: dataPtr is null but dataSize is > 0" << endl;
		return;
	}

    if(parameters.overrideMemoryRestrictions && parameters.resultSaveType != SAVE_TYPE_LIST){
		cout << "[Init] Error: Can't override memory restrictions when saving as SAVE_TYPE_ALL" << endl;
		return;
	}

    idxSteps.push_back(1);
    for (i = 1; i < parameters.D; i++) {
        idxSteps.push_back(idxSteps[i - 1] * limits[i-1].N);
	}

    for (i = 0; i < parameters.D; i++) {
		limits[i].step = abs(limits[i].upperLimit - limits[i].lowerLimit) / limits[i].N;
	}

	#ifdef DBG_DATA
        for(i=0; i < parameters.D; i++){
			printf("Dimension %d: Low=%lf, High=%lf, Step=%lf, N=%u, idxSteps=%llu\n", i, limits[i].lowerLimit, limits[i].upperLimit, limits[i].step, limits[i].N, idxSteps[i]);
		}
	#endif

	totalReceived = 0;
	totalSent = 0;
    totalElements = (unsigned long long)(idxSteps[parameters.D - 1]) * (unsigned long long)(limits[parameters.D - 1].N);

	if(rank == 0){
        if(! (parameters.benchmark)){
            if(parameters.resultSaveType == SAVE_TYPE_ALL){
                if(parameters.saveFile.size()){
					// No saveFile given, save everything in memory
					finalResults = new RESULT_TYPE[totalElements];		// Uninitialized
				}else{
					// Open save file
                    saveFile = open(parameters.saveFile.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
					if(saveFile == -1){
						fatal("open failed");
					}

					// Enlarge the file
					if(ftruncate(saveFile, totalElements * sizeof(RESULT_TYPE)) == -1){
						fatal("ftruncate failed");
					}

					// Map the save file in memory
					finalResults = (RESULT_TYPE*) mmap(nullptr, totalElements * sizeof(RESULT_TYPE), PROT_WRITE, MAP_SHARED, saveFile, 0);
					if(finalResults == MAP_FAILED){
						fatal("mmap failed");
					}

					#ifdef DBG_MEMORY
						printf("[Init] finalResults: 0x%lx\n", finalResults);
					#endif
				}
			}// else listResults will be allocated through realloc when they are needed
		}

        if (parameters.batchSize == 0)
            parameters.batchSize = totalElements;
	}

	valid = true;
}

ParallelFramework::~ParallelFramework() {
	if(rank == 0){
        if(parameters.saveFile.size() == 0){
			delete [] finalResults;
		}else{
			// Unmap the save file
			munmap(finalResults, totalElements * sizeof(RESULT_TYPE));

			// Close the file
			close(saveFile);
		}
	}
	valid = false;
}

bool ParallelFramework::isValid() const {
	return valid;
}

const RESULT_TYPE* ParallelFramework::getResults() const {
	if(rank != 0){
        throw std::runtime_error("Error: Results can only be fetched by the master process. Are you the master process?\n");
	}

    if(parameters.resultSaveType != SAVE_TYPE_ALL){
        throw std::runtime_error("Error: Can't get all results when resultSaveType is not SAVE_TYPE_ALL\n");
	}

	return finalResults;
}

const std::vector<std::vector<DATA_TYPE>>& ParallelFramework::getList() const {
    if(rank != 0)
        throw std::runtime_error("Error: Results can only be fetched by the master process. Are you the master process?\n");

    if(parameters.resultSaveType != SAVE_TYPE_LIST){
        throw std::runtime_error("Error: Can't get list results when resultSaveType is not SAVE_TYPE_LIST\n");
	}

	return listResults;
}

void ParallelFramework::getIndicesFromPoint(DATA_TYPE* point, unsigned long* dst) const {
	unsigned int i;

    for (i = 0; i < parameters.D; i++) {
		if (point[i] < limits[i].lowerLimit || point[i] >= limits[i].upperLimit) {
            throw std::invalid_argument("Result query for out-of-bounds point\n");
		}

		// Calculate the steps for dimension i
		dst[i] = (int) round(abs(limits[i].lowerLimit - point[i]) / limits[i].step);		// TODO: 1.9999997 will round to 2, verify correctness
	}
}

unsigned long ParallelFramework::getIndexFromIndices(unsigned long* pointIdx) const {
	unsigned int i;
	unsigned long index = 0;

    for (i = 0; i < parameters.D; i++) {
		// Increase index by i*(index-steps for this dimension)
		index += pointIdx[i] * idxSteps[i];
	}

	return index;
}

unsigned long ParallelFramework::getIndexFromPoint(DATA_TYPE* point) const {
    unsigned long indices[parameters.D];
	unsigned long index;

	getIndicesFromPoint(point, indices);
	index = getIndexFromIndices(indices);

	return index;
}

void ParallelFramework::getPointFromIndex(unsigned long index, DATA_TYPE* result) const {
    for(int i=parameters.D - 1; i>=0; i--){
		int currentIndex = index / idxSteps[i];
		result[i] = limits[i].lowerLimit + currentIndex*limits[i].step;

		index = index % idxSteps[i];
	}
}

int ParallelFramework::getRank() const {
    return rank;
}
