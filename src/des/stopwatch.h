#pragma once

#include <sys/time.h>

class Stopwatch {
private:
    timespec t1, t2;

public:
    void start();
    void stop();
    float getNsec() const;
    float getMsec() const;
};
