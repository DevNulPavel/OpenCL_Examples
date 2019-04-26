#ifndef _TIMER_H_
#define _TIMER_H_
/**
 * \file Timer.h
 * \brief A timer class that provides a cross platform timer for use
 * in timing code progress with a high degree of accuracy.
 */
#ifdef _WIN32
/**
 * \typedef __int64 i64
 * \brief Maps the windows 64 bit integer to a uniform name
 */
#if defined(__MINGW64__) || defined(__MINGW32__)
typedef long long i64;
#else
typedef __int64 i64;
#endif
#else
/**
 * \typedef long long i64
 * \brief Maps the linux 64 bit integer to a uniform name
 */
typedef long long i64;
#endif

/**
 * \class Timer
 * \brief Counter that provides a fairly accurate timing mechanism for both
 * windows and linux. This timer is used extensively in all the samples.
 */
class Timer {

public:
    Timer();
    ~Timer();
    void Start(void);
    void Stop(void);
    void Reset(void);
    double GetElapsedTime(void);

private:

    i64 _freq;
    i64 _clocks;
    i64 _start;
};

#endif // _TIMER_H_

