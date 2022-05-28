#pragma once

#include "scheduler.h"

namespace desf {

template<typename node_id_t>
class ConstantScheduler : public Scheduler<node_id_t> {
public:
    ConstantScheduler(size_t k)
        : m_constant(k)
    {}

    size_t getBatchSize(node_id_t) override {
        return m_constant;
    }

private:
    size_t m_constant;
};

}
