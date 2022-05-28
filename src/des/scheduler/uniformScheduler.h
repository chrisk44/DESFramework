#pragma once

#include "scheduler.h"

namespace desf {

template<typename node_id_t>
class UniformScheduler : public Scheduler<node_id_t> {
public:
    void init(const DesConfig& config, const std::vector<node_id_t>& nodes, const std::map<node_id_t, size_t>& maxBatchSizes) override {
        Scheduler<node_id_t>::init(config, nodes, maxBatchSizes);
        m_numOfNodes = nodes.size();
    }

    size_t getBatchSize(node_id_t) override {
        return m_batchSize;
    }

protected:
    virtual void onWorkSet(AssignedWork work) override {
        size_t numOfElements = work.numOfElements;
        m_batchSize = numOfElements / m_numOfNodes;

        while(m_batchSize * m_numOfNodes < numOfElements) {
            size_t diff = (numOfElements - m_batchSize);

            m_batchSize += std::max((size_t) 1, diff / m_numOfNodes);

            #ifdef DBG_SNH
                this->log("Recalculated batch size %lu from diff = %lu\n", m_batchSize, diff);
            #endif
        }
    }

private:
    size_t m_numOfNodes;
    size_t m_batchSize;
};

}
