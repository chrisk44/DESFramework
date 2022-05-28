#pragma once

#include <atomic>
#include <cmath>
#include <map>
#include <mpi.h>
#include <set>
#include <stdarg.h>
#include <vector>

#include "../types.h"

namespace desf {

template<typename node_id_t>
class Scheduler {
public:
    Scheduler() {
        MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    };
    virtual ~Scheduler() {};

    void setWork(AssignedWork work) {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        m_work = work;
        m_assignedElements = 0;

        this->onWorkSet(work);
    }

    size_t getAssignedElements() { return m_assignedElements; }

    virtual void init(const DesConfig&, const std::vector<node_id_t>&, const std::map<node_id_t, size_t>& maxBatchSizes) {
        #ifdef DBG_SCHEDULE
            log("Initializing...");
        #endif
        m_maxBatchSizes = maxBatchSizes;
        #ifdef DBG_SCHEDULE
            for(const auto& pair : m_maxBatchSizes) {
                log("Node %d has max batch size = %lu", (int) pair.first, pair.second);
            }
        #endif
    }
    virtual void finalize() {
        #ifdef DBG_SCHEDULE
            log("Finalizing");
        #endif
    }

    virtual size_t getBatchSize(node_id_t) = 0;

    virtual AssignedWork getNextBatch(node_id_t node) {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);

        const auto& work = getWork();
        decltype(work.startPoint) lastPoint = work.startPoint + work.numOfElements - 1;

        auto batchSize = this->getBatchSize(node);

        #ifdef DBG_SCHEDULE
            log("Got %lu elements for node %d", batchSize, (int) node);
        #endif

        AssignedWork toReturn;
        toReturn.numOfElements = std::min(batchSize, m_maxBatchSizes[node]);
        toReturn.startPoint = work.startPoint + m_assignedElements;

        #ifdef DBG_SCHEDULE
            log("Assigning %lu elements to node %d", toReturn.numOfElements, (int) node);
        #endif

        if(toReturn.startPoint > lastPoint || toReturn.numOfElements == 0) {
            #ifdef DBG_SCHEDULE
                log("Node %d requested data but all has been assigned, returning 0", (int) node);
            #endif
            return AssignedWork(0, 0);
        }

        if(toReturn.startPoint + toReturn.numOfElements - 1 > lastPoint) {
            toReturn.numOfElements = lastPoint - toReturn.startPoint + 1;
        }

        m_assignedElements += toReturn.numOfElements;

        return toReturn;
    }

    virtual void onNodeStarted(node_id_t) {}
    virtual void onNodeFinished(node_id_t, size_t, float) {}

protected:
    virtual void onWorkSet(AssignedWork) {}

    const AssignedWork& getWork() const {
        return m_work;
    }

    void log(const char* text, ...) {
        static thread_local char buf[LOG_BUFFER_SIZE];

        va_list args;
        va_start(args, text);
        vsnprintf(buf, sizeof(buf), text, args);
        va_end(args);

        std::string spaces = m_rank == 0 ? "" : "    ";
        printf("[%d] %sScheduler: %s\n", m_rank, spaces.c_str(), buf);
    }

private:
    AssignedWork m_work;

    std::mutex m_mutex;
    size_t m_assignedElements;

    std::map<node_id_t, size_t> m_maxBatchSizes;
    int m_rank;
};

}
