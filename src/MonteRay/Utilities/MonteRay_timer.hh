#ifndef CPUTIMER_HH_
#define CPUTIMER_HH_

#include <time.h>

namespace MonteRay {

class tripleTime{
public:
    double gpuTime;
    double cpuTime;
    double totalTime;
};

class cpuTimer {
public:
    cpuTimer() :
        startTime( timespec() ),
        stopTime(  timespec() )
    {}

    ~cpuTimer(){};

    //static double ElapsedTime(timespec start, timespec end);
    static double ElapsedTime(timespec start, timespec end) {
        double seconds = end.tv_sec - start.tv_sec;
        seconds += double(end.tv_nsec - start.tv_nsec) / 1000000000;
        return seconds;
    }

    double getTime(void) const { return ElapsedTime(startTime, stopTime); }

    void start(void) {clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &startTime);}
    void stop(void) {clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stopTime);}

private:
    timespec startTime;
    timespec stopTime;

};

} /* namespace MonteRay */

#endif /* CPUTIMER_HH_ */
