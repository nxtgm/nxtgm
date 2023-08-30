#include <nxtgm/utils/timer.hpp>
#include <stdexcept>

namespace nxtgm
{

void Timer::start()
{
    start_ = clock_type::now();
    running_ = true;
    duration_ = duration_type::zero();
}

void Timer::pause()
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
void Timer::resume()
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

typename Timer::duration_type Timer::elapsed() const
{

    return duration_ + (clock_type::now() - start_);
}

}; // namespace nxtgm
