#pragma once

#include "scheduler.h"

namespace desf {

template<typename node_id_t>
class HRSLSScheduler : public Scheduler<node_id_t> {
public:
    HRSLSScheduler(size_t maxBatchSize)
        : m_hrslsMaxBatchSize(maxBatchSize),
          m_minTimeToAdjust(0.f),
          m_minElementsToAdjust(0),
          m_useAverage(false)
    {}

    void setMinTimeToAdjustRatios(float time) { m_minTimeToAdjust = time; }
    void setMinElementsToAdjustRatios(size_t count) { m_minElementsToAdjust = count; }
    void setUseAverageRatio() { m_useAverage = true; }
    void setUseLatestRatio() { m_useAverage = false; }

    virtual void init(const DesConfig& config, const std::vector<node_id_t>& nodes, const std::map<node_id_t, size_t>& maxBatchSizes) override {
        Scheduler<node_id_t>::init(config, nodes, maxBatchSizes);

        m_ratios.clear();
        m_avgRatios.clear();
        m_scoreSums.clear();
        m_lastScores.clear();
        m_assignmentsPerNode.clear();
        for(const auto& node : nodes) {
            m_ratios[node] = 1.f / (float) nodes.size();
            m_avgRatios[node] = m_ratios[node];
            m_scoreSums[node] = 0.f;
            m_assignmentsPerNode[node] = 0;
            // Do NOT initialize m_lastScores with anything, it needs to contain only valid scores
        }
    }

    virtual size_t getBatchSize(node_id_t node) override {
        const auto& work = this->getWork();

        float ratio = m_useAverage ? m_avgRatios[node] : m_ratios[node];
        size_t batchSize = ceil(work.numOfElements * ratio);

        // Limit batch size by our max batch size
        return std::min(batchSize, m_hrslsMaxBatchSize);
    }

    virtual void onNodeFinished(node_id_t node, size_t elements, float time) override {
        if(elements < m_minElementsToAdjust || time < m_minTimeToAdjust) {
            #ifdef DBG_RATIO
                this->log("Skipping ratio adjustment due to low time or low number of elements (%lu < %lu || %.2f < %.2f)", elements, m_minElementsToAdjust, time, m_minTimeToAdjust);
            #endif
            return;
        }

        float score = (float) elements / time;
        m_lastScores[node] = score;
        m_scoreSums[node] += score;
        m_assignmentsPerNode[node]++;

        #ifdef DBG_RATIO
            this->log("New score for node %d: %.2f (%lu elements, %.2fms)", (int) node, m_lastScores[node], elements, time);
        #endif

        recalculateRatios();

        #ifdef DBG_RATIO
            std::string ratios;
            std::string avgRatios;
            for(const auto& pair : m_ratios){
                char tmp[1024];
                sprintf(tmp, "[%d - %.2f]", (int) pair.first, pair.second);
                ratios += tmp;

                sprintf(tmp, "[%d - %.2f]", (int) pair.first, m_avgRatios[pair.first]);
                avgRatios += tmp;
            }
            this->log("New ratios: %s", ratios.c_str());
            this->log("Avg ratios: %s", avgRatios.c_str());
        #endif
    }

protected:
    virtual void recalculateRatios() {
        // If any node does not have a score yet, skip recalculation
        if(m_lastScores.size() != m_ratios.size()) {
            #ifdef DBG_RATIO
                this->log("Skipping ratio adjustment due to not enough scores");
            #endif
            return;
        }

        // Calculate the scores' sum
        float totalScoreLatest = 0.f;
        float totalScoreAvg = 0.f;
        for(const auto& pair : m_lastScores) {
            totalScoreLatest += pair.second;
            if(std::isnan(m_scoreSums[pair.first])) throw std::runtime_error("Score sum for node " + std::to_string(pair.first) + " is NaN");
            if(m_assignmentsPerNode[pair.first] == 0) throw std::runtime_error("Node " + std::to_string(pair.first) + " has 0 assignments");
            totalScoreAvg += m_scoreSums[pair.first] / m_assignmentsPerNode[pair.first];
        }

        if(std::isnan(totalScoreAvg)) throw std::runtime_error("totalScoreAvg is NaN");
        if(totalScoreAvg == 0.f) throw std::runtime_error("totalScoreAvg is 0.0");

        // Calculate a new ratio for each node
        for(auto& pair : m_ratios) {
            const auto& node = pair.first;
            pair.second = m_lastScores[node] / totalScoreLatest;
            m_avgRatios[node] = (m_scoreSums[node] / m_assignmentsPerNode[node]) / totalScoreAvg;
        }
    }

    const size_t m_hrslsMaxBatchSize;

    // Ratio adjustment parameters
    float m_minTimeToAdjust;
    size_t m_minElementsToAdjust;
    bool m_useAverage;

    std::map<node_id_t, float> m_ratios;
    std::map<node_id_t, float> m_avgRatios;
    std::map<node_id_t, float> m_scoreSums;
    std::map<node_id_t, float> m_lastScores;
    std::map<node_id_t, int> m_assignmentsPerNode;
};

}
