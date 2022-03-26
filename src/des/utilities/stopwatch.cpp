#include "stopwatch.h"

#include <time.h>

void Stopwatch::start()
{
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
}

void Stopwatch::stop()
{
    clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
}

float Stopwatch::getNsec() const
{
    timespec difference;
    if ((t2.tv_nsec - t1.tv_nsec)<0) {
        difference.tv_sec = t2.tv_sec - t1.tv_sec - 1;
        difference.tv_nsec = 1000000000 + t2.tv_nsec - t1.tv_nsec;
    } else {
        difference.tv_sec = t2.tv_sec - t1.tv_sec;
        difference.tv_nsec = t2.tv_nsec - t1.tv_nsec;
    }

    return difference.tv_sec * 1000000000 + difference.tv_nsec;
}

float Stopwatch::getMsec() const
{
    return getNsec() / 1000000.0;
}
