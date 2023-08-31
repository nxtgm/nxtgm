#pragma once
#include <chrono>

namespace nxtgm
{

class Timer
{
  public:
    using clock_type = std::chrono::high_resolution_clock;
    using time_point = std::chrono::time_point<clock_type>;
    using duration_type = std::chrono::duration<double>;

    Timer() = default;

    void start();
    void pause();
    void resume();

    duration_type elapsed() const;

  private:
    time_point start_;
    duration_type duration_;
    bool running_;
};

}; // namespace nxtgm
