#include <chrono>

namespace  nxtgm
{
    class AutoStartedTimer
    {   
    public:
        using clock_type = std::chrono::high_resolution_clock;
        using time_point = std::chrono::time_point<clock_type>;
        using duration_type = std::chrono::duration<double>;

    private:

        struct RAIIPauseResume{
            inline RAIIPauseResume(AutoStartedTimer & timer):timer_(timer){
                timer_.pause();
            }
            inline ~RAIIPauseResume(){timer_.resume();}
            AutoStartedTimer & timer_;
        };



        void start()
        {
            start_ = clock_type::now();
            running_ = true;
            duration_ = duration_type::zero();
        }

        void pause()
        {
            if(running_){
                duration_ += clock_type::now() - start_;
                running_ = false;
            }
            else{
                throw std::runtime_error("Timer already paused");
            }
        }
        void resume()
        {
            if(!running_){
                start_ = clock_type::now();
                running_ = true;
            }
            else{
                throw std::runtime_error("Timer already running");
            }
        }


    public:


        AutoStartedTimer(){
            this->start();
        }

        template <typename Fn>
        std::invoke_result_t<Fn> paused_call(Fn && fn)
        {
            RAIIPauseResume pause_resume(*this);
            return fn();
        }


        
        duration_type elapsed()const
        {

            return duration_ + (clock_type::now() - start_);
        }





    private:
        time_point start_;
        duration_type duration_;
        bool running_;
    };
};