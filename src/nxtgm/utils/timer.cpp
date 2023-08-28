#include <nxtgm/utils/timer.hpp>
#include <stdexcept>

namespace nxtgm
{

AutoStartedTimer::RAIIPauseResume::RAIIPauseResume(AutoStartedTimer &timer)
    : timer_(timer)
{
    timer_.pause();
}
AutoStartedTimer::RAIIPauseResume::~RAIIPauseResume()
{
    timer_.resume();
}

void AutoStartedTimer::start()
{
    start_ = clock_type::now();
    running_ = true;
    duration_ = duration_type::zero();
}

void AutoStartedTimer::pause()
{
    if (running_)
    {
        duration_ += clock_type::now() - start_;
        running_ = false;
    }
    else
    {
        throw std::runtime_error("Timer already paused");
    }
}
void AutoStartedTimer::resume()
{
    if (!running_)
    {
        start_ = clock_type::now();
        running_ = true;
    }
    else
    {
        throw std::runtime_error("Timer already running");
    }
}

AutoStartedTimer::AutoStartedTimer()
{
    this->start();
}

typename AutoStartedTimer::duration_type AutoStartedTimer::elapsed() const
{

    return duration_ + (clock_type::now() - start_);
}

}; // namespace nxtgm
