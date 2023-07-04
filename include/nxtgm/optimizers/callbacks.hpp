#ifndef NXTGM_OPTIMIZERS_OPTIMIZER_BASE_HPP
#define NXTGM_OPTIMIZERS_OPTIMIZER_BASE_HPP

// for inf
#include <limits>

// tuple
#include <tuple>

// visitor impl
#include <chrono>
#include <iostream>
#include <fmt/core.h>
#include <fmt/chrono.h>



#include <nxtgm/optimizers/callback_base.hpp>


namespace nxtgm
{




    template<class OPTIMIZER_BASE_TYPE>
    class ReporterCallback : public ReporterCallbackBase<OPTIMIZER_BASE_TYPE> {
    public:
        using base_type = ReporterCallbackBase<OPTIMIZER_BASE_TYPE>;
        using optimizer_base_type = typename base_type::optimizer_base_type;
        using time_point_type = std::chrono::high_resolution_clock::time_point;

        ReporterCallback(const optimizer_base_type * optimizer) : base_type(optimizer) {
        }

        void begin() override {
            t_begin_ = std::chrono::high_resolution_clock::now();
            t_last = t_begin_;
            this->print();
        }
        void end() override {
            this->print();
        }

        bool report() override {
            this->print();
            ++iteration_;
            return true;
        }

    private:
        void print(){
            const auto now = std::chrono::high_resolution_clock::now();
            const auto dt_total = now - t_begin_;
            const auto dt_last = now - t_last;
            t_last = now;

            const auto current = this->optimizer()->current_solution_value();
            const auto best = this->optimizer()->best_solution_value();
            const auto lower_bound = this->optimizer()->lower_bound();

            std::cout<<fmt::format("{:5} {:<10} {:<10}", iteration_,dt_last, dt_total);
            std::cout<<fmt::format("{:<12.3e} ", lower_bound);
            std::cout<<fmt::format("{:<12.3e} {:<6} {:<12.3e}   ", current.energy() , current.is_feasible(), current.how_violated());
            std::cout<<fmt::format("{:<12.3e} {:<6} {:<12.3e}\n", best.energy() , best.is_feasible(), best.how_violated());

        }

        time_point_type t_begin_;
        time_point_type t_last;
        std::size_t iteration_;
    };






} //nxtgm::optimizers


#endif // NXTGM_OPTIMIZERS_OPTIMIZER_BASE_HPP
