#include <limits.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <mpi.h>

#include "desf.h"

namespace desf {

DesFramework::DesFramework(const DesConfig& config)
    : m_config(config),
      m_saveFile(-1),
      m_finalResults(nullptr),
      m_totalSent(0),
      m_totalReceived(0),
      m_totalElements(0),
      m_rank(-1)
{
    if(config.handleMPI){
		// Initialize MPI
		MPI_Init(nullptr, nullptr);
	}
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

    if(config.limits.size() != config.model.D)
        throw std::invalid_argument("The limits vector must have a size equal to config.D");

    if(config.batchSize <= 0)
        throw std::invalid_argument("Batch size must be a positive value");

    for (unsigned int i = 0; i < config.model.D; i++) {
        if (config.limits[i].lowerLimit > config.limits[i].upperLimit)
            throw std::invalid_argument("Lower limit for dimension " + std::to_string(i) + " can't be higher than upper limit");

        if (config.limits[i].N == 0)
            throw std::invalid_argument("N for dimension " + std::to_string(i) + " must be > 0");
    }

    if(config.model.dataPtr == nullptr && config.model.dataSize > 0)
        throw std::invalid_argument("dataPtr is null but dataSize is > 0");

    if(config.output.overrideMemoryRestrictions && config.resultSaveType != SAVE_TYPE_LIST)
        throw std::invalid_argument("Can't override memory restrictions when saving as SAVE_TYPE_ALL");

    if(m_rank != 0){
        if(config.processingType != PROCESSING_TYPE_GPU){
            if(config.cpu.forwardModel == nullptr)
                throw std::invalid_argument("CPU forward model function is nullptr");

            if(config.cpu.objective == nullptr)
                throw std::invalid_argument("CPU objective function is nullptr");
        }

        if(config.processingType != PROCESSING_TYPE_CPU){
            if(config.gpu.size() == 0)
                throw std::invalid_argument("GPU forward model functions not provided");

            for(const auto& pair : config.gpu) {
                if(pair.second.forwardModel == nullptr)
                    throw std::invalid_argument("GPU forward model function for GPU " + std::to_string(pair.first) + " is nullptr");

                if(pair.second.objective == nullptr)
                    throw std::invalid_argument("GPU objective function for GPU " + std::to_string(pair.first) + " is nullptr");
            }
        }
    }

    for (unsigned int i = 0; i < m_config.model.D; i++) {
        m_config.limits[i].step = abs(m_config.limits[i].upperLimit - m_config.limits[i].lowerLimit) / m_config.limits[i].N;

        m_idxSteps.push_back(i==0 ? 1 : m_idxSteps[i - 1] * m_config.limits[i-1].N);
    }

    #ifdef DBG_DATA
        for(unsigned int i=0; i < m_config.model.D; i++){
            printf("Dimension %u: Low=%lf, High=%lf, Step=%lf, N=%u, m_idxSteps=%llu\n", i, m_config.limits[i].lowerLimit, m_config.limits[i].upperLimit, m_config.limits[i].step, m_config.limits[i].N, m_idxSteps[i]);
        }
    #endif

    #ifdef DBG_MEMORY
        printf("CPU  : Forward model is @ %p, objective is @ %p\n", m_config.cpu.forwardModel, m_config.cpu.objective);
        for(const auto& pair : m_config.gpu)
            printf("GPU %d: Forward model is @ %p, objective is @ %p\n", pair.first, pair.second.forwardModel, pair.second.objective);
    #endif

    m_totalReceived = 0;
    m_totalSent = 0;
    m_totalElements = (unsigned long long)(m_idxSteps[m_config.model.D - 1]) * (unsigned long long)(m_config.limits[m_config.model.D - 1].N);

    if(m_rank == 0){
        if(! (m_config.benchmark)){
            if(m_config.resultSaveType == SAVE_TYPE_ALL){
                if(m_config.output.saveFile.size()){
					// No saveFile given, save everything in memory
                    m_finalResults = new RESULT_TYPE[m_totalElements];		// Uninitialized
				}else{
					// Open save file
                    m_saveFile = open(m_config.output.saveFile.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
                    if(m_saveFile == -1){
                        fatal("open failed");
					}

					// Enlarge the file
                    if(ftruncate(m_saveFile, m_totalElements * sizeof(RESULT_TYPE)) == -1){
						fatal("ftruncate failed");
                    }

					// Map the save file in memory
                    m_finalResults = (RESULT_TYPE*) mmap(nullptr, m_totalElements * sizeof(RESULT_TYPE), PROT_WRITE, MAP_SHARED, m_saveFile, 0);
                    if(m_finalResults == MAP_FAILED){
						fatal("mmap failed");
					}

					#ifdef DBG_MEMORY
                        printf("[Init] finalResults: %p\n", m_finalResults);
					#endif
				}
            }// else listResults will be dynamically allocated when needed
        }
	}
}

DesFramework::~DesFramework() {
    if(m_rank == 0){
        if(m_config.output.saveFile.size() == 0) {
            delete [] m_finalResults;
        } else if(m_saveFile != -1) {
			// Unmap the save file
            munmap(m_finalResults, m_totalElements * sizeof(RESULT_TYPE));

			// Close the file
            close(m_saveFile);
		}
	}

    if(m_config.handleMPI)
        MPI_Finalize();
}

void DesFramework::run() {
    if(m_rank == 0){
        if(m_config.printProgress) printf("[%d] Master process starting\n", m_rank);
        masterProcess();
        if(m_config.printProgress) printf("[%d] Master process finished\n", m_rank);
    }else{
        if(m_config.printProgress) printf("[%d] Slave process starting\n", m_rank);
        slaveProcess();
        if(m_config.printProgress) printf("[%d] Slave process finished\n", m_rank);
    }
}

const RESULT_TYPE* DesFramework::getResults() const {
    if(m_rank != 0)
        throw std::runtime_error("Error: Results can only be fetched by the master process. Are you the master process?\n");

    if(m_config.resultSaveType != SAVE_TYPE_ALL)
        throw std::runtime_error("Error: Can't get all results when resultSaveType is not SAVE_TYPE_ALL\n");

    return m_finalResults;
}

const std::vector<std::vector<DATA_TYPE>>& DesFramework::getList() const {
    if(m_rank != 0)
        throw std::runtime_error("Error: Results can only be fetched by the master process. Are you the master process?\n");

    if(m_config.resultSaveType != SAVE_TYPE_LIST)
        throw std::runtime_error("Error: Can't get list results when resultSaveType is not SAVE_TYPE_LIST\n");

    return m_listResults;
}

void DesFramework::getIndicesFromPoint(DATA_TYPE* point, unsigned long* dst) const {
	unsigned int i;

    for (i = 0; i < m_config.model.D; i++) {
        if (point[i] < m_config.limits[i].lowerLimit || point[i] >= m_config.limits[i].upperLimit)
            throw std::invalid_argument("Result query for out-of-bounds point\n");

		// Calculate the steps for dimension i
        dst[i] = (int) round(abs(m_config.limits[i].lowerLimit - point[i]) / m_config.limits[i].step);		// TODO: 1.9999997 will round to 2, verify correctness
	}
}

unsigned long DesFramework::getIndexFromIndices(unsigned long* pointIdx) const {
	unsigned int i;
	unsigned long index = 0;

    for (i = 0; i < m_config.model.D; i++) {
		// Increase index by i*(index-steps for this dimension)
        index += pointIdx[i] * m_idxSteps[i];
	}

	return index;
}

unsigned long DesFramework::getIndexFromPoint(DATA_TYPE* point) const {
    unsigned long indices[m_config.model.D];
	unsigned long index;

	getIndicesFromPoint(point, indices);
	index = getIndexFromIndices(indices);

	return index;
}

void DesFramework::getPointFromIndex(unsigned long index, DATA_TYPE* result) const {
    for(int i=m_config.model.D - 1; i>=0; i--){
        int currentIndex = index / m_idxSteps[i];
        result[i] = m_config.limits[i].lowerLimit + currentIndex*m_config.limits[i].step;

        index = index % m_idxSteps[i];
	}
}

int DesFramework::getRank() const {
    return m_rank;
}

}
