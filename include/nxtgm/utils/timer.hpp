#pragma once
#include <chrono>

namespace nxtgm
{
class AutoStartedTimer
{
  public:
    using clock_type = std::chrono::high_resolution_clock;
    using time_point = std::chrono::time_point<clock_type>;
    using duration_type = std::chrono::duration<double>;

  private:
    struct RAIIPauseResume
    {
        RAIIPauseResume(AutoStartedTimer &timer);
        ~RAIIPauseResume();
        AutoStartedTimer &timer_;
    };

    void start();
    void pause();
    void resume();

  public:
    AutoStartedTimer();

    template <typename Fn>
    std::invoke_result_t<Fn> paused_call(Fn &&fn)
    {
        RAIIPauseResume pause_resume(*this);
        return fn();
    }

    duration_type elapsed() const;

  private:
    time_point start_;
    duration_type duration_;
    bool running_;
};
}; // namespace nxtgm
