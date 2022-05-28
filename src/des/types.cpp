#include "types.h"
#include "scheduler/scheduler.h"

namespace desf {

DesConfig::DesConfig(bool deleteSchedulersWhenDestructed)
    : m_deleteSchedulers(deleteSchedulersWhenDestructed)
{}

DesConfig::~DesConfig()
{
    if(m_deleteSchedulers) {
        delete intraNodeScheduler;
        delete interNodeScheduler;
    }
}

}
