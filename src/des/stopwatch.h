#pragma once

#include <sys/time.h>

namespace desf {

class Stopwatch {
private:
    timespec t1, t2;

public:
    void start();
    void stop();
    float getNsec() const;
    float getMsec() const;
};

}
