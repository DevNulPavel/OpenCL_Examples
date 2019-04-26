
#include "Timer.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#include <time.h>
#endif

Timer::Timer() : _clocks(0), _start(0)
{

#ifdef _WIN32
    QueryPerformanceFrequency((LARGE_INTEGER *)&_freq);
#else
    _freq = 1000;
#endif

}

Timer::~Timer()
{
    // EMPTY!
}

void
Timer::Start(void)
{

#ifdef _WIN32
    QueryPerformanceCounter((LARGE_INTEGER *)&_start);
#else
    struct timespec s;
    clock_gettime( CLOCK_REALTIME, &s );
    _start = (i64)s.tv_sec * 1e9 + (i64)s.tv_nsec;
#endif

}

void
Timer::Stop(void)
{
    i64 n;

#ifdef _WIN32
    QueryPerformanceCounter((LARGE_INTEGER *)&n);
#else
    struct timespec s;
    clock_gettime( CLOCK_REALTIME, &s );
    n = (i64)s.tv_sec * 1e9 + (i64)s.tv_nsec;
#endif

    n -= _start;
    _start = 0;
    _clocks += n;
}

void
Timer::Reset(void)
{

    _clocks = 0;
}

double
Timer::GetElapsedTime(void)
{
#if _WIN32
    return (double)_clocks / (double) _freq;
#else
    return (double)_clocks / (double) 1e9;
#endif

}

