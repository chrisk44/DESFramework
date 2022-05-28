#pragma once

#include "hrslsScheduler.h"

namespace desf {

template<typename node_id_t>
class HRDLSScheduler : public HRSLSScheduler<node_id_t> {
public:
    HRDLSScheduler(size_t constantBatchSize, size_t maxBatchSize)
        : HRSLSScheduler<node_id_t> (maxBatchSize),
          m_constantBatchSize(constantBatchSize),
          m_slowStartSteps(0)
    {}

    void setSlowStart(int steps, size_t base) {
        m_slowStartSteps = steps;
        m_slowStartBase = base;
    }
    void unsetSlowStart() { m_slowStartSteps = 0; }

    size_t getBatchSize(node_id_t node) override {
        size_t batchSize = HRSLSScheduler<node_id_t>::getBatchSize(node);
        int prevAssignments = this->m_assignmentsPerNode.at(node);
        if(prevAssignments < m_slowStartSteps) {
            // Limit batch size by the slow-start limit
            batchSize = std::min(batchSize, (size_t) ceil(m_slowStartBase * pow(2, prevAssignments)));
        }

        return batchSize;
    }

private:
    size_t m_constantBatchSize;

    // Slow start
    int m_slowStartSteps;
    size_t m_slowStartBase;
};

}
