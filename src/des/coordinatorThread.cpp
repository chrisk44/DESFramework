#include "desf.h"

#include "coordinatorThread.h"
#include "scheduler/scheduler.h"
#include "utilities.h"

#include <stdarg.h>

namespace desf {

CoordinatorThread::CoordinatorThread(const DesConfig& config, std::vector<ComputeThread>& threads)
    : m_config(config),
      m_threads(threads),
      m_maxBatchSize(calculateMaxBatchSize(config)),
      m_maxCpuBatchSize(calculateMaxCpuBatchSize(config))
{
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

    std::vector<ComputeThreadID> threadIds;
    std::map<ComputeThreadID, size_t> maxBatchSizes;
    for(const auto& thread : m_threads){
        threadIds.push_back(thread.getId());
        maxBatchSizes[thread.getId()] = thread.getType() == CPU ? m_maxCpuBatchSize : thread.getGpuMaxBatchSize();
    }
    m_config.intraNodeScheduler->init(m_config, threadIds, maxBatchSizes);
}

CoordinatorThread::~CoordinatorThread()
{
    m_config.intraNodeScheduler->finalize();
}

void CoordinatorThread::log(const char *text, ...) {
    static thread_local char buf[LOG_BUFFER_SIZE];

    va_list args;
    va_start(args, text);
    vsnprintf(buf, sizeof(buf), text, args);
    va_end(args);

    printf("[%d]     Coordinator: %s\n", m_rank, buf);
}

void CoordinatorThread::run(){
    WorkDispatcher workDispatcher = [&](ComputeThreadID id){
        auto work = m_config.intraNodeScheduler->getNextBatch(id);
        #ifdef DBG_DATA
            log("Assigning batch with %lu elements starting from %lu to %d", work.numOfElements, work.startPoint, id);
        #endif
        return work;
    };

	#ifdef DBG_TIME
		Stopwatch sw;
		float time_data, time_assign, time_start, time_wait, time_scores, time_results;
    #endif

    #ifdef DBG_DATA
        log("Max batch size: %lu", m_maxBatchSize);
    #elif defined(DBG_MPI_STEPS)
        log("Sending max batch size = %lu", m_maxBatchSize);
    #endif
    DesFramework::sendMaxBatchSize(m_maxBatchSize);

	while(true){
		// Send READY signal to master
		#ifdef DBG_MPI_STEPS
            log("Sending READY...");
		#endif
		#ifdef DBG_TIME
			sw.start();
		#endif

        // Get a batch of data from master
        DesFramework::sendReadyRequest();
        AssignedWork work = DesFramework::receiveWorkFromMaster();

		#ifdef DBG_DATA
            log("Received %lu elements starting from %lu", work.numOfElements, work.startPoint);
		#endif
		#ifdef DBG_MPI_STEPS
            log("Received %lu elements", work.numOfElements);
		#endif

		// If no elements, break
		if(work.numOfElements == 0)
			break;

		// Make sure we have enough allocated memory for localResults
        size_t neededBytes;
        if(m_config.resultSaveType == SAVE_TYPE_ALL) {
            neededBytes = work.numOfElements * sizeof(RESULT_TYPE);
        } else {
            neededBytes = work.numOfElements * m_config.model.D * sizeof(DATA_TYPE);
            if(m_config.output.overrideMemoryRestrictions) {
                #ifdef DBG_MEMORY
                    log("Limiting needed memory from %lu bytes to %lu bytes", neededBytes, getMaxCPUBytes());
                #endif
                neededBytes = std::min(neededBytes, getMaxCPUBytes());
            }
        }
        if(neededBytes > m_results.capacity() * sizeof(RESULT_TYPE)){
            #ifdef DBG_MEMORY
                log("Allocating more memory for results: %lu -> %lu MB", (m_results.size() * sizeof(RESULT_TYPE))/(1024*1024), neededBytes/(1024*1024));
            #endif

            unsigned long newSize = neededBytes / sizeof(RESULT_TYPE);
            m_results.resize(newSize);
            if(m_results.size() < newSize){
                fatal("Can't allocate memory for results");
			}
		}

		#ifdef DBG_TIME
			sw.stop();
			time_data = sw.getMsec();
			sw.start();
        #endif

        m_globalFirst      = work.startPoint;
        m_globalLast       = work.startPoint + work.numOfElements - 1;
        m_globalBatchStart = work.startPoint;

		#ifdef DBG_DATA
            log("Got job with globalFirst = %lu, globalLast = %lu", m_globalFirst, m_globalLast);
		#endif

        m_config.intraNodeScheduler->setWork(work);

		#ifdef DBG_TIME
			sw.stop();
			time_assign = sw.getMsec();
			sw.start();
		#endif

		// Reset the global listIndex counter
        m_listIndex = 0;

		// Start all the worker threads
        for(auto& cti : m_threads){
            m_config.intraNodeScheduler->onNodeStarted(cti.getId());
            cti.dispatch(workDispatcher, m_results.data(), work.startPoint, &m_listIndex);
		}

		#ifdef DBG_TIME
			sw.stop();
			time_start = sw.getMsec();
			sw.start();
		#endif

		#ifdef DBG_MPI_STEPS
            log("Waiting for results from %lu threads...", m_threads.size());
		#endif
		// Wait for all worker threads to finish their work
        for(auto& ct : m_threads){
            ct.wait();
		}

		#ifdef DBG_TIME
			sw.stop();
			time_wait = sw.getMsec();
		#endif

		// Sanity check
        size_t totalCalculated = 0;
        for(const auto& ct : m_threads)
            totalCalculated += ct.getLastCalculatedElements();

		if(totalCalculated != work.numOfElements){
            log("Shit happened: Total calculated elements are %lu but %lu were assigned to this slave", totalCalculated, work.numOfElements);

            for(const auto& ct : m_threads){
                log("Thread %d calculated %lu elements", ct.getId(), ct.getLastCalculatedElements());
            }

			exit(-123);
		}

        #ifdef DBG_RESULTS
            if(m_config.resultSaveType == SAVE_TYPE_LIST){
                log("Found %u results", m_listIndex / m_config.model.D);
            }
        #endif
        #ifdef DBG_RESULTS_RAW
            if(m_config.resultSaveType == SAVE_TYPE_ALL){
                std::string str;
                str += "[ ";
                for(unsigned long i=0; i<work.numOfElements; i++){
                    char tmp[64];
                    sprintf(tmp, "%f ", m_results[i]);
                    str += tmp;
				}
                str += "]";
                log("%s", str.c_str());
            }else{
                std::string str;
                for(int i=0; i<m_listIndex; i+=m_config.model.D){
                    str += "[ ";
                    for(unsigned int j=0; j<m_config.model.D; j++){
                        char tmp[64];
                        sprintf(tmp, "%f ", ((DATA_TYPE *) m_results.data())[i + j]);
                        str += tmp;
                    }
                    str += " ]";
                }
                log("Results (m_listIndex = %d): %s", m_listIndex, str.c_str());
			}
		#endif

        #ifdef DBG_TIME
            sw.start();
        #endif

        for(const auto& ct : m_threads) {
            m_config.intraNodeScheduler->onNodeFinished(ct.getId(), ct.getLastCalculatedElements(), ct.getLastRunTime());

            if(m_config.benchmark){
                log("Thread %d time: %f ms", ct.getId(), ct.getLastRunTime());
            }
        }

        #ifdef DBG_TIME
            sw.stop();
            time_scores = sw.getMsec();
            sw.start();
        #endif

		// Send all results to master
		#ifdef DBG_MPI_STEPS
            log("Sending data to master...");
		#endif

		#ifdef DBG_TIME
			sw.start();
		#endif

        if(m_config.resultSaveType == SAVE_TYPE_ALL){
            DesFramework::sendResults(m_results.data(), work.numOfElements);
        }else{
            DesFramework::sendListResults((DATA_TYPE*) m_results.data(), m_listIndex / m_config.model.D, m_config.model.D);
		}

		#ifdef DBG_TIME
			sw.stop();
			time_results = sw.getMsec();

            log("Benchmark:");
            log("Time for receiving data: %f ms", time_data);
            log("Time for assign: %f ms", time_assign);
            log("Time for start: %f ms", time_start);
            log("Time for wait: %f ms", time_wait);
            log("Time for scores: %f ms", time_scores);
            log("Time for results: %f ms", time_results);
		#endif
    }
}

unsigned long CoordinatorThread::calculateMaxBatchSize(const DesConfig& config)
{
    if(config.output.overrideMemoryRestrictions){
        return std::numeric_limits<size_t>::max();
    }

    unsigned long maxBatchSize;
    if(config.processingType == PROCESSING_TYPE_CPU)
        maxBatchSize = getMaxCPUBytes();
    else if(config.processingType == PROCESSING_TYPE_GPU)
        maxBatchSize = getMaxGPUBytes();
    else
        maxBatchSize = std::min(getMaxCPUBytes(), getMaxGPUBytes());

    // maxBatchSize contains the value in bytes, so divide it according to resultSaveType to convert it to actual batch size
    if(config.resultSaveType == SAVE_TYPE_ALL)
        maxBatchSize /= sizeof(RESULT_TYPE);
    else
        maxBatchSize /= config.model.D * sizeof(DATA_TYPE);

    // If we are saving a list, the max number of elements we might want to send is maxBatchSize * D, so limit the batch size
    // so that the max number of elements is INT_MAX
    if(config.resultSaveType == SAVE_TYPE_LIST && (unsigned long) (maxBatchSize*config.model.D) > (unsigned long) INT_MAX){
        maxBatchSize = (INT_MAX - config.model.D) / config.model.D;
    }
    // If we are saving all of the results, the max number of elements is maxBatchSize itself, so limit it to INT_MAX
    else if(config.resultSaveType == SAVE_TYPE_ALL && (unsigned long) maxBatchSize > (unsigned long) INT_MAX){
        maxBatchSize = INT_MAX;
    }

    return maxBatchSize;
}

unsigned long CoordinatorThread::calculateMaxCpuBatchSize(const DesConfig& config)
{
    if(config.output.overrideMemoryRestrictions) {
        return std::numeric_limits<size_t>::max();
    } else if(config.resultSaveType == SAVE_TYPE_ALL) {
        return getMaxCPUBytes() / sizeof(RESULT_TYPE);
    } else  {
        return getMaxCPUBytes() / (config.model.D * sizeof(DATA_TYPE));
    }
}

}
