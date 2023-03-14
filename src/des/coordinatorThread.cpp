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
    #ifdef DBG_TIME
        Stopwatch sw;
        sw.start();
    #endif

    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

    std::vector<ComputeThreadID> threadIds;
    std::map<ComputeThreadID, size_t> maxBatchSizes;
    for(const auto& thread : m_threads){
        threadIds.push_back(thread.getId());
        maxBatchSizes[thread.getId()] = thread.getType() == CPU ? m_maxCpuBatchSize : thread.getGpuMaxBatchSize();
    }
    m_config.intraNodeScheduler->init(m_config, threadIds, maxBatchSizes);

    #ifdef DBG_TIME
        sw.stop();
        log("Time for initialization: %f ms", sw.getMsec());
    #endif
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

    printf("%s [%d]     Coordinator: %s\n", getTimeString().c_str(), m_rank, buf);
}

void CoordinatorThread::run(){
    #ifdef DBG_DATA
        log("Max batch size: %lu", m_maxBatchSize);
    #elif defined(DBG_MPI_STEPS)
        log("Sending max batch size = %lu", m_maxBatchSize);
    #endif

    #ifdef DBG_TIME
        Stopwatch sw;
        sw.start();
    #endif

    DesFramework::sendMaxBatchSize(m_maxBatchSize);

    #ifdef DBG_TIME
        sw.stop();
        log("Time to send max batch size: %f ms", sw.getMsec());
    #endif

    ComputeEnvironment env;
	while(true){
        #ifdef DBG_TIME
            float time_data, time_env, time_start, time_wait, time_scores, time_results;
            std::atomic_long time_assign;   // nanoseconds
            sw.start();
        #endif

        WorkDispatcher workDispatcher = [&](ComputeThreadID id){
            #ifdef DBG_TIME
                Stopwatch tempSw;
                tempSw.start();
            #endif
            auto work = m_config.intraNodeScheduler->getNextBatch(id);
            #ifdef DBG_TIME
                tempSw.stop();
                time_assign.fetch_add(tempSw.getNsec());
            #endif
            #ifdef DBG_DATA
                log("Assigning batch with %lu elements starting from %lu to %d", work.numOfElements, work.startPoint, id);
            #endif
            return work;
        };

		// Send READY signal to master
		#ifdef DBG_MPI_STEPS
            log("Sending READY...");
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

        #ifdef DBG_TIME
            sw.stop();
            time_data = sw.getMsec();
            sw.start();
        #endif

		// If no elements, break
		if(work.numOfElements == 0)
			break;

        if(m_config.resultSaveType == SAVE_TYPE_ALL) {
            env.setup(workDispatcher, work.numOfElements, work.startPoint);
        } else {
            env.setup(workDispatcher);
        }

		#ifdef DBG_TIME
			sw.stop();
            time_env = sw.getMsec();
			sw.start();
        #endif

        m_config.intraNodeScheduler->setWork(work);

		#ifdef DBG_TIME
			sw.stop();
            time_assign.store(sw.getNsec());
			sw.start();
		#endif

		// Start all the worker threads
        for(auto& cti : m_threads){
            cti.dispatch(env);
            m_config.intraNodeScheduler->onNodeStarted(cti.getId());
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
                size_t count;
                env.getListResults(&count);
                log("Found %u results", count);
            }
        #endif
        #ifdef DBG_RESULTS_RAW
            size_t listCount;
            size_t* listResults = m_config.resultSaveType == SAVE_TYPE_LIST ? env.getListResults(&listCount) : nullptr;

            std::string str;
            str += "[ ";
            if(m_config.resultSaveType == SAVE_TYPE_ALL){
                for(unsigned long i=0; i<work.numOfElements; i++){
                    char tmp[64];
                    sprintf(tmp, "%f ", *env.getAddrForIndex(work.startPoint + 1));
                    str += tmp;
                }
            }else{
                for(unsigned long i=0; i<listCount; i++){
                    char tmp[64];
                    sprintf(tmp, "%lu ", listResults[i]);
                    str += tmp;
                }
            }
            str += "]";
            log("Results: %s", str.c_str());
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
            DesFramework::sendResults(env.getAddrForIndex(0), work.numOfElements);
        }else{
            size_t count;
            size_t* addr = env.getListResults(&count);
            DesFramework::sendListResults(addr, count);
		}

		#ifdef DBG_TIME
			sw.stop();
			time_results = sw.getMsec();

            log("Benchmark:");
            log("Time for receiving data: %f ms", time_data);
            log("Time to setup environment: %f ms", time_env);
            log("Time for assignments: %f ms", time_assign.load() / 1000.f);
            log("Time for starting compute threads: %f ms", time_start);
            log("Time for all threads to finish: %f ms", time_wait);
            log("Time for scheduler to be updated: %f ms", time_scores);
            log("Time for sending results: %f ms", time_results);
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
    maxBatchSize /= config.resultSaveType == SAVE_TYPE_ALL ? sizeof(RESULT_TYPE) : sizeof(size_t);

    maxBatchSize = std::min(maxBatchSize, (size_t) INT_MAX);

    return maxBatchSize;
}

unsigned long CoordinatorThread::calculateMaxCpuBatchSize(const DesConfig& config)
{
    if(config.output.overrideMemoryRestrictions) {
        return std::numeric_limits<size_t>::max();
    } else if(config.resultSaveType == SAVE_TYPE_ALL) {
        return getMaxCPUBytes() / sizeof(RESULT_TYPE);
    } else {
        return getMaxCPUBytes() / sizeof(size_t);
    }
}

}
