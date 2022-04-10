#include <limits.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <mpi.h>

#include "framework.h"

ParallelFramework::ParallelFramework(bool initMPI)
    : m_saveFile(-1),
      m_finalResults(nullptr),
      m_valid(false),
      m_totalSent(0),
      m_totalReceived(0),
      m_totalElements(0),
      m_rank(-1)
{
	if(initMPI){
		// Initialize MPI
		MPI_Init(nullptr, nullptr);
	}
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
}

void ParallelFramework::init(const std::vector<Limit>& limits, const ParallelFrameworkParameters& parameters){
    if(limits.size() != parameters.model.D)
        throw std::invalid_argument("The limits vector must have a size equal to parameters.D");

    for (unsigned int i = 0; i < parameters.model.D; i++) {
        if (limits[i].lowerLimit > limits[i].upperLimit)
            throw std::invalid_argument("Lower limit for dimension " + std::to_string(i) + " can't be higher than upper limit");

        if (limits[i].N == 0)
            throw std::invalid_argument("N for dimension " + std::to_string(i) + " must be > 0");
    }

    if(parameters.model.dataPtr == nullptr && parameters.model.dataSize > 0)
        throw std::invalid_argument("dataPtr is null but dataSize is > 0");

    if(parameters.output.overrideMemoryRestrictions && parameters.resultSaveType != SAVE_TYPE_LIST)
        throw std::invalid_argument("Can't override memory restrictions when saving as SAVE_TYPE_ALL");

    m_parameters = parameters;
    m_limits = limits;

    for (unsigned int i = 0; i < parameters.model.D; i++) {
        m_limits[i].step = abs(m_limits[i].upperLimit - m_limits[i].lowerLimit) / m_limits[i].N;

        m_idxSteps.push_back(i==0 ? 1 : m_idxSteps[i - 1] * m_limits[i-1].N);
    }

    #ifdef DBG_DATA
        for(unsigned int i=0; i < parameters.model.D; i++){
            printf("Dimension %u: Low=%lf, High=%lf, Step=%lf, N=%u, m_idxSteps=%llu\n", i, m_limits[i].lowerLimit, m_limits[i].upperLimit, m_limits[i].step, m_limits[i].N, m_idxSteps[i]);
        }
    #endif

    m_totalReceived = 0;
    m_totalSent = 0;
    m_totalElements = (unsigned long long)(m_idxSteps[parameters.model.D - 1]) * (unsigned long long)(limits[parameters.model.D - 1].N);

    if(m_rank == 0){
        if(! (m_parameters.benchmark)){
            if(m_parameters.resultSaveType == SAVE_TYPE_ALL){
                if(m_parameters.output.saveFile.size()){
					// No saveFile given, save everything in memory
                    m_finalResults = new RESULT_TYPE[m_totalElements];		// Uninitialized
				}else{
					// Open save file
                    m_saveFile = open(m_parameters.output.saveFile.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
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

        if (m_parameters.batchSize == 0)
            m_parameters.batchSize = m_totalElements;
	}

    m_valid = true;
}

ParallelFramework::~ParallelFramework() {
    if(m_rank == 0){
        if(m_parameters.output.saveFile.size() == 0) {
            delete [] m_finalResults;
        } else if(m_saveFile != -1) {
			// Unmap the save file
            munmap(m_finalResults, m_totalElements * sizeof(RESULT_TYPE));

			// Close the file
            close(m_saveFile);
		}
	}
    m_valid = false;
}

bool ParallelFramework::isValid() const {
    return m_valid;
}

const RESULT_TYPE* ParallelFramework::getResults() const {
    if(m_rank != 0)
        throw std::runtime_error("Error: Results can only be fetched by the master process. Are you the master process?\n");

    if(m_parameters.resultSaveType != SAVE_TYPE_ALL)
        throw std::runtime_error("Error: Can't get all results when resultSaveType is not SAVE_TYPE_ALL\n");

    return m_finalResults;
}

const std::vector<std::vector<DATA_TYPE>>& ParallelFramework::getList() const {
    if(m_rank != 0)
        throw std::runtime_error("Error: Results can only be fetched by the master process. Are you the master process?\n");

    if(m_parameters.resultSaveType != SAVE_TYPE_LIST)
        throw std::runtime_error("Error: Can't get list results when resultSaveType is not SAVE_TYPE_LIST\n");

    return m_listResults;
}

void ParallelFramework::getIndicesFromPoint(DATA_TYPE* point, unsigned long* dst) const {
	unsigned int i;

    for (i = 0; i < m_parameters.model.D; i++) {
        if (point[i] < m_limits[i].lowerLimit || point[i] >= m_limits[i].upperLimit)
            throw std::invalid_argument("Result query for out-of-bounds point\n");

		// Calculate the steps for dimension i
        dst[i] = (int) round(abs(m_limits[i].lowerLimit - point[i]) / m_limits[i].step);		// TODO: 1.9999997 will round to 2, verify correctness
	}
}

unsigned long ParallelFramework::getIndexFromIndices(unsigned long* pointIdx) const {
	unsigned int i;
	unsigned long index = 0;

    for (i = 0; i < m_parameters.model.D; i++) {
		// Increase index by i*(index-steps for this dimension)
        index += pointIdx[i] * m_idxSteps[i];
	}

	return index;
}

unsigned long ParallelFramework::getIndexFromPoint(DATA_TYPE* point) const {
    unsigned long indices[m_parameters.model.D];
	unsigned long index;

	getIndicesFromPoint(point, indices);
	index = getIndexFromIndices(indices);

	return index;
}

void ParallelFramework::getPointFromIndex(unsigned long index, DATA_TYPE* result) const {
    for(int i=m_parameters.model.D - 1; i>=0; i--){
        int currentIndex = index / m_idxSteps[i];
        result[i] = m_limits[i].lowerLimit + currentIndex*m_limits[i].step;

        index = index % m_idxSteps[i];
	}
}

int ParallelFramework::getRank() const {
    return m_rank;
}
