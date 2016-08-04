#include "cpuTimer.h"

namespace MonteRay {

cpuTimer::cpuTimer() :
		startTime( timespec() ),
		stopTime(  timespec() )
{

}

cpuTimer::~cpuTimer() {

}

//double
//cpuTimer::ElapsedTime(timespec start, timespec end)
//{
//	double seconds = end.tv_sec - start.tv_sec;
//	seconds += double(end.tv_nsec - start.tv_nsec) / 1000000000;
//
//	return seconds;
//}

} /* namespace MonteRay */
